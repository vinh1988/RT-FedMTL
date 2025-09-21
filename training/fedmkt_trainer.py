#!/usr/bin/env python3
"""
FedMKT Trainer for DistilBART ↔ MobileBART Classification
Implements federated knowledge transfer training for 20News classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BartTokenizer, get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import json
import os
from dataclasses import dataclass
import copy

from models.bart_classification import DistilBARTClassifier, MobileBARTClassifier, create_lora_config
from data.news20_dataset import News20Dataset

logger = logging.getLogger(__name__)


@dataclass
class FedMKTTrainingConfig:
    """Configuration for FedMKT training"""
    # Model configurations
    distilbart_model_name: str = "facebook/bart-base"
    mobilebart_model_name: str = "facebook/bart-base"
    num_labels: int = 20
    max_length: int = 512
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Knowledge distillation
    distill_temperature: float = 4.0
    kd_alpha: float = 0.7
    distill_loss_type: str = "kl"  # "kl" or "ce"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # Output
    output_dir: str = "./outputs/fedmkt_20news"


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for classification"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss


class FedMKTTrainer:
    """Federated Model Knowledge Transfer Trainer"""
    
    def __init__(self, config: FedMKTTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.central_model = None
        self.client_models = {}
        
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(config.distilbart_model_name)
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=config.distill_temperature,
            alpha=config.kd_alpha
        )
        
        # Training history
        self.training_history = {
            "central_model": [],
            "client_models": {},
            "knowledge_transfer": []
        }
        
        logger.info(f"FedMKT Trainer initialized on device: {self.device}")
    
    def create_central_model(self) -> nn.Module:
        """Create central DistilBART model"""
        logger.info("Creating central DistilBART model...")
        
        lora_config = None
        if self.config.use_lora:
            lora_config = create_lora_config(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
            )
        
        model = DistilBARTClassifier(
            model_name=self.config.distilbart_model_name,
            num_labels=self.config.num_labels,
            max_length=self.config.max_length
        )
        
        if lora_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, lora_config)
        
        model.to(self.device)
        self.central_model = model
        
        logger.info(f"Central model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def create_client_model(self, client_id: int) -> nn.Module:
        """Create client MobileBART model"""
        logger.info(f"Creating client {client_id} MobileBART model...")
        
        lora_config = None
        if self.config.use_lora:
            lora_config = create_lora_config(
                r=self.config.lora_r // 2,  # Smaller LoRA for clients
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj"]
            )
        
        model = MobileBARTClassifier(
            model_name=self.config.mobilebart_model_name,
            num_labels=self.config.num_labels,
            max_length=self.config.max_length // 2
        )
        
        if lora_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, lora_config)
        
        model.to(self.device)
        self.client_models[client_id] = model
        
        logger.info(f"Client {client_id} model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def setup_optimizer(self, model: nn.Module, learning_rate: float) -> Tuple[torch.optim.Optimizer, Any]:
        """Setup optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer, None
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(data_loader)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def train_local_epoch(self, model: nn.Module, data_loader: DataLoader, 
                         optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train model for one local epoch"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "acc": f"{correct_predictions / total_predictions:.4f}"
            })
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(data_loader)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def knowledge_distillation_step(self, client_model: nn.Module, 
                                   central_logits: torch.Tensor,
                                   public_data_loader: DataLoader,
                                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Knowledge distillation step"""
        client_model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(public_data_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Get teacher logits (from central model)
            with torch.no_grad():
                teacher_outputs = self.central_model(input_ids, attention_mask)
                teacher_logits = teacher_outputs["logits"]
            
            # Get student logits (from client model)
            student_outputs = client_model(input_ids, attention_mask)
            student_logits = student_outputs["logits"]
            
            # Compute knowledge distillation loss
            kd_loss = self.kd_loss(student_logits, teacher_logits, labels)
            
            kd_loss = kd_loss / self.config.gradient_accumulation_steps
            kd_loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += kd_loss.item() * self.config.gradient_accumulation_steps
            
            predictions = torch.argmax(student_logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(public_data_loader)
        
        return {
            "kd_loss": avg_loss,
            "accuracy": accuracy
        }
    
    def aggregate_client_knowledge(self, client_logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate knowledge from client models"""
        # Simple averaging of logits
        stacked_logits = torch.stack(client_logits_list)
        aggregated_logits = torch.mean(stacked_logits, dim=0)
        return aggregated_logits
    
    def federated_training_round(self, client_data_loaders: Dict[int, Dict[str, DataLoader]],
                                public_data_loader: DataLoader, round_idx: int) -> Dict[str, Any]:
        """One round of federated training"""
        logger.info(f"Starting federated training round {round_idx + 1}")
        
        round_results = {
            "round": round_idx + 1,
            "central_model": {},
            "client_models": {},
            "knowledge_transfer": {}
        }
        
        # Phase 1: Central model generates knowledge for clients
        logger.info("Phase 1: Central model → Client models knowledge transfer")
        self.central_model.eval()
        
        with torch.no_grad():
            central_logits_list = []
            for batch in public_data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.central_model(input_ids, attention_mask)
                central_logits_list.append(outputs["logits"])
            
            central_logits = torch.cat(central_logits_list, dim=0)
        
        # Distribute knowledge to clients
        client_kd_results = {}
        for client_id, client_model in self.client_models.items():
            if client_id in client_data_loaders:
                logger.info(f"Training client {client_id} with central knowledge")
                
                optimizer, _ = self.setup_optimizer(client_model, self.config.learning_rate)
                
                # Knowledge distillation
                kd_results = self.knowledge_distillation_step(
                    client_model, central_logits, public_data_loader, optimizer
                )
                client_kd_results[client_id] = kd_results
                
                # Local training on private data
                logger.info(f"Local training for client {client_id}")
                local_results = self.train_local_epoch(
                    client_model, client_data_loaders[client_id]["train"], optimizer, round_idx + 1
                )
                
                round_results["client_models"][client_id] = {
                    "knowledge_distillation": kd_results,
                    "local_training": local_results
                }
        
        # Phase 2: Aggregate client knowledge to central model
        logger.info("Phase 2: Client models → Central model knowledge aggregation")
        client_logits_list = []
        
        for client_id, client_model in self.client_models.items():
            client_model.eval()
            with torch.no_grad():
                client_logits_batch = []
                for batch in public_data_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    outputs = client_model(input_ids, attention_mask)
                    client_logits_batch.append(outputs["logits"])
                
                client_logits = torch.cat(client_logits_batch, dim=0)
                client_logits_list.append(client_logits)
        
        # Aggregate client knowledge
        aggregated_logits = self.aggregate_client_knowledge(client_logits_list)
        
        # Update central model with aggregated knowledge
        logger.info("Updating central model with aggregated knowledge")
        optimizer, _ = self.setup_optimizer(self.central_model, self.config.learning_rate)
        
        self.central_model.train()
        total_kd_loss = 0
        
        for batch in public_data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Get central model predictions
            central_outputs = self.central_model(input_ids, attention_mask)
            central_logits = central_outputs["logits"]
            
            # Compute loss with aggregated client knowledge
            start_idx = input_ids.size(0) * 0  # Simplified indexing
            end_idx = start_idx + input_ids.size(0)
            target_logits = aggregated_logits[start_idx:end_idx]
            
            kd_loss = self.kd_loss(central_logits, target_logits, labels)
            
            kd_loss = kd_loss / self.config.gradient_accumulation_steps
            kd_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_kd_loss += kd_loss.item() * self.config.gradient_accumulation_steps
        
        avg_kd_loss = total_kd_loss / len(public_data_loader)
        
        round_results["central_model"]["knowledge_aggregation"] = {
            "kd_loss": avg_kd_loss
        }
        
        # Evaluate models
        logger.info("Evaluating models...")
        if "val" in client_data_loaders[list(client_data_loaders.keys())[0]]:
            val_loader = client_data_loaders[list(client_data_loaders.keys())[0]]["val"]
            
            # Evaluate central model
            central_eval = self.evaluate_model(self.central_model, val_loader)
            round_results["central_model"]["evaluation"] = central_eval
            
            # Evaluate client models
            for client_id, client_model in self.client_models.items():
                client_eval = self.evaluate_model(client_model, val_loader)
                round_results["client_models"][client_id]["evaluation"] = client_eval
        
        logger.info(f"Completed federated training round {round_idx + 1}")
        return round_results
    
    def train(self, client_data_loaders: Dict[int, Dict[str, DataLoader]], 
              public_data_loader: DataLoader) -> Dict[str, Any]:
        """Main federated training loop"""
        logger.info("Starting federated training...")
        
        # Create models if not already created
        if self.central_model is None:
            self.create_central_model()
        
        if not self.client_models:
            for client_id in client_data_loaders.keys():
                self.create_client_model(client_id)
        
        # Training loop
        for round_idx in range(self.config.num_epochs):
            round_results = self.federated_training_round(
                client_data_loaders, public_data_loader, round_idx
            )
            
            self.training_history["central_model"].append(round_results["central_model"])
            for client_id, client_results in round_results["client_models"].items():
                if client_id not in self.training_history["client_models"]:
                    self.training_history["client_models"][client_id] = []
                self.training_history["client_models"][client_id].append(client_results)
            
            # Log results
            logger.info(f"Round {round_idx + 1} Results:")
            if "evaluation" in round_results["central_model"]:
                central_eval = round_results["central_model"]["evaluation"]
                logger.info(f"  Central Model - Loss: {central_eval['loss']:.4f}, Accuracy: {central_eval['accuracy']:.4f}")
            
            for client_id, client_results in round_results["client_models"].items():
                if "evaluation" in client_results:
                    client_eval = client_results["evaluation"]
                    logger.info(f"  Client {client_id} - Loss: {client_eval['loss']:.4f}, Accuracy: {client_eval['accuracy']:.4f}")
        
        # Save training history
        self.save_training_history()
        
        logger.info("Federated training completed!")
        return self.training_history
    
    def save_training_history(self):
        """Save training history to file"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save central model
        central_path = os.path.join(self.config.output_dir, "central_model")
        torch.save(self.central_model.state_dict(), os.path.join(central_path, "pytorch_model.bin"))
        
        # Save client models
        for client_id, client_model in self.client_models.items():
            client_path = os.path.join(self.config.output_dir, f"client_model_{client_id}")
            torch.save(client_model.state_dict(), os.path.join(client_path, "pytorch_model.bin"))
        
        logger.info("Models saved successfully")


def test_fedmkt_trainer():
    """Test function for FedMKT trainer"""
    print("Testing FedMKT Trainer...")
    
    # Create config
    config = FedMKTTrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Create trainer
    trainer = FedMKTTrainer(config)
    
    # Create dummy data
    from torch.utils.data import TensorDataset, DataLoader
    
    batch_size = 4
    seq_length = 128
    num_samples = 32
    
    # Create dummy datasets
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    labels = torch.randint(0, 20, (num_samples,))
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create client data loaders
    client_data_loaders = {
        0: {"train": data_loader, "val": data_loader},
        1: {"train": data_loader, "val": data_loader},
        2: {"train": data_loader, "val": data_loader}
    }
    
    try:
        # Test model creation
        central_model = trainer.create_central_model()
        client_model_0 = trainer.create_client_model(0)
        client_model_1 = trainer.create_client_model(1)
        client_model_2 = trainer.create_client_model(2)
        
        print("Models created successfully!")
        
        # Test one training round
        print("Testing one federated training round...")
        round_results = trainer.federated_training_round(client_data_loaders, data_loader, 0)
        
        print("Training round completed successfully!")
        print(f"Round results keys: {round_results.keys()}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    
    print("FedMKT Trainer test completed successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the trainer
    test_fedmkt_trainer()
