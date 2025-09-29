#!/usr/bin/env python3
"""
Knowledge Distillation + LoRA Federated Learning
- Uses LoRA for parameter efficiency (within same architecture)
- Uses knowledge distillation for cross-architecture learning
- Solves the incompatible layers problem properly
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDistillationConfig:
    """Configuration for knowledge distillation federated learning"""
    global_model_name: str = "bert-base-uncased"
    client_model_name: str = "prajjwal1/bert-tiny"
    num_clients: int = 3
    num_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_sequence_length: int = 128
    
    # Knowledge distillation parameters
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7  # Weight for distillation loss
    
    # LoRA parameters (for efficiency within same architecture)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class LoRALayer(nn.Module):
    """LoRA layer for parameter efficiency"""
    
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return lora_out


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, original_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, r, alpha, dropout)
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for teacher-student learning"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss (soft targets)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Task loss (hard targets)
        task_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        return total_loss, distillation_loss, task_loss


def add_lora_to_linear_layers(model, config: KnowledgeDistillationConfig, target_modules: List[str] = None):
    """Add LoRA to specific linear layers"""
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
    
    lora_modules = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            lora_linear = LoRALinear(module, config.lora_r, config.lora_alpha, config.lora_dropout)
            
            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, child_name, lora_linear)
            else:
                setattr(model, child_name, lora_linear)
            
            lora_modules[name] = lora_linear
            logger.info(f"Added LoRA to {name}: {module.in_features} -> {module.out_features}")
    
    return lora_modules


class FederatedModelWithLoRA(nn.Module):
    """Model with LoRA for federated learning"""
    
    def __init__(self, model_name: str, config: KnowledgeDistillationConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        # Add LoRA to attention layers only (for efficiency)
        self.lora_modules = add_lora_to_linear_layers(self.bert, config, ["query", "key", "value"])
        
        logger.info(f"Model initialized: {model_name}")
        logger.info(f"Added LoRA to {len(self.lora_modules)} modules")
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_hidden_states=False):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states
        )
        
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        result = {'loss': loss, 'logits': logits}
        if return_hidden_states:
            result['hidden_states'] = outputs.hidden_states
            result['pooler_output'] = outputs.pooler_output
        
        return result


class KnowledgeDistillationServer:
    """Server that uses knowledge distillation for federated learning"""
    
    def __init__(self, config: KnowledgeDistillationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Global teacher model (BERT-base)
        self.global_model = FederatedModelWithLoRA(config.global_model_name, config)
        self.global_model.to(self.device)
        
        logger.info(f"Knowledge Distillation Server initialized: {config.global_model_name}")
    
    def get_teacher_knowledge(self, dataloader) -> Dict[str, torch.Tensor]:
        """Extract knowledge (logits) from global teacher model"""
        self.global_model.eval()
        
        all_logits = []
        all_hidden_states = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.global_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=True
                )
                
                all_logits.append(outputs['logits'].cpu())
                all_hidden_states.append(outputs['pooler_output'].cpu())
        
        return {
            'logits': torch.cat(all_logits, dim=0),
            'hidden_states': torch.cat(all_hidden_states, dim=0)
        }
    
    def get_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Get LoRA parameters from global model"""
        lora_params = {}
        
        for name, module in self.global_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_params[f"{name}.lora_A"] = module.lora.lora_A.detach().cpu().numpy()
                lora_params[f"{name}.lora_B"] = module.lora.lora_B.detach().cpu().numpy()
        
        return lora_params
    
    def set_lora_parameters(self, lora_params: Dict[str, np.ndarray]):
        """Set LoRA parameters in global model"""
        for name, module in self.global_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_a_name = f"{name}.lora_A"
                lora_b_name = f"{name}.lora_B"
                
                if lora_a_name in lora_params:
                    module.lora.lora_A.data = torch.tensor(lora_params[lora_a_name]).to(self.device)
                if lora_b_name in lora_params:
                    module.lora.lora_B.data = torch.tensor(lora_params[lora_b_name]).to(self.device)
    
    def aggregate_lora_parameters(self, client_lora_params: List[Dict[str, np.ndarray]], 
                                 client_weights: List[float]) -> Dict[str, np.ndarray]:
        """Aggregate LoRA parameters from clients with same architecture"""
        if not client_lora_params:
            return {}
        
        aggregated_params = {}
        total_weight = sum(client_weights)
        
        # Only aggregate parameters that exist in global model
        global_lora_params = self.get_lora_parameters()
        
        for param_name in global_lora_params.keys():
            # Check if all clients have this parameter with same shape
            compatible_clients = []
            compatible_weights = []
            
            for client_params, weight in zip(client_lora_params, client_weights):
                if (param_name in client_params and 
                    client_params[param_name].shape == global_lora_params[param_name].shape):
                    compatible_clients.append(client_params[param_name])
                    compatible_weights.append(weight)
            
            if compatible_clients:
                # Weighted average of compatible parameters
                total_compatible_weight = sum(compatible_weights)
                aggregated_params[param_name] = np.zeros_like(compatible_clients[0])
                
                for client_param, weight in zip(compatible_clients, compatible_weights):
                    normalized_weight = weight / total_compatible_weight
                    aggregated_params[param_name] += normalized_weight * client_param
        
        logger.info(f"Aggregated {len(aggregated_params)} LoRA parameters from compatible clients")
        return aggregated_params
    
    def evaluate_global_model(self, test_dataset) -> Dict[str, float]:
        """Evaluate global model"""
        self.global_model.eval()
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.global_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(test_loader)
        
        return {'loss': avg_loss, 'accuracy': accuracy}


class KnowledgeDistillationClient:
    """Client that learns from teacher through knowledge distillation"""
    
    def __init__(self, client_id: int, config: KnowledgeDistillationConfig, dataset):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Student model (Tiny-BERT)
        self.model = FederatedModelWithLoRA(config.client_model_name, config)
        self.model.to(self.device)
        
        # Knowledge distillation loss
        self.kd_loss = KnowledgeDistillationLoss(config.distillation_temperature, config.distillation_alpha)
        
        # Only train LoRA parameters + classifier
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'lora' in name or 'classifier' in name:
                trainable_params.append(param)
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        
        logger.info(f"KD Client {client_id} initialized with {len(dataset)} samples")
        logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params_count:,} ({100*trainable_params_count/total_params:.1f}%)")
    
    def local_training_with_distillation(self, teacher_knowledge: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Train with knowledge distillation from teacher"""
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student forward pass
                student_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Calculate loss
                if teacher_knowledge is not None and batch_idx < len(teacher_knowledge['logits'])//self.config.batch_size:
                    # Use knowledge distillation
                    start_idx = batch_idx * self.config.batch_size
                    end_idx = start_idx + input_ids.size(0)
                    teacher_logits = teacher_knowledge['logits'][start_idx:end_idx].to(self.device)
                    
                    loss, kd_loss, task_loss = self.kd_loss(
                        student_outputs['logits'],
                        teacher_logits,
                        labels
                    )
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()
                else:
                    # Regular training without teacher
                    loss = student_outputs['loss']
                    total_task_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(student_outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / (len(dataloader) * self.config.local_epochs)
        avg_kd_loss = total_kd_loss / (len(dataloader) * self.config.local_epochs)
        avg_task_loss = total_task_loss / (len(dataloader) * self.config.local_epochs)
        accuracy = correct_predictions / total_samples
        
        logger.info(f"KD Client {self.client_id} training completed:")
        logger.info(f"  Loss: {avg_loss:.4f}, KD Loss: {avg_kd_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'loss': avg_loss, 
            'kd_loss': avg_kd_loss,
            'task_loss': avg_task_loss,
            'accuracy': accuracy, 
            'num_samples': len(self.dataset)
        }
    
    def get_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Get client's LoRA parameters"""
        lora_params = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_params[f"{name}.lora_A"] = module.lora.lora_A.detach().cpu().numpy()
                lora_params[f"{name}.lora_B"] = module.lora.lora_B.detach().cpu().numpy()
        
        return lora_params


# Reuse SimpleDataset from previous implementations
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dummy_datasets(num_clients: int = 3, samples_per_client: int = 100):
    """Create dummy datasets for each client"""
    datasets = []
    
    base_texts = [
        ["This is a positive sentiment", "Great product, love it", "Amazing experience", "Wonderful service"],
        ["This is negative feedback", "Terrible quality", "Very disappointed", "Poor customer service"],
        ["Neutral statement here", "Average product quality", "Standard service", "Regular experience"]
    ]
    
    for i in range(num_clients):
        texts = []
        labels = []
        
        base_text_set = base_texts[i % len(base_texts)]
        
        for j in range(samples_per_client):
            text = f"{base_text_set[j % len(base_text_set)]} - sample {j} from client {i}"
            label = j % 2
            
            texts.append(text)
            labels.append(label)
        
        datasets.append((texts, labels))
        logger.info(f"Created dataset for client {i} with {len(texts)} samples")
    
    return datasets


def run_knowledge_distillation_federated_learning():
    """Run knowledge distillation based federated learning"""
    config = KnowledgeDistillationConfig(
        num_clients=3, 
        num_rounds=3, 
        local_epochs=2, 
        lora_r=8,
        distillation_alpha=0.7
    )
    
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION + LoRA FEDERATED LEARNING DEMO")
    print("=" * 80)
    print(f"Global Teacher Model: {config.global_model_name}")
    print(f"Client Student Model: {config.client_model_name}")
    print(f"LoRA Rank: {config.lora_r}, Alpha: {config.lora_alpha}")
    print(f"Distillation Temperature: {config.distillation_temperature}")
    print(f"Distillation Alpha: {config.distillation_alpha}")
    print(f"Number of Clients: {config.num_clients}")
    print(f"Number of Rounds: {config.num_rounds}")
    print("=" * 80)
    
    # Initialize server
    server = KnowledgeDistillationServer(config)
    
    # Create client datasets
    client_datasets_data = create_dummy_datasets(config.num_clients)
    
    # Initialize clients
    clients = []
    tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
    
    for i in range(config.num_clients):
        texts, labels = client_datasets_data[i]
        dataset = SimpleDataset(texts, labels, tokenizer, config.max_sequence_length)
        client = KnowledgeDistillationClient(i, config, dataset)
        clients.append(client)
    
    # Create test dataset
    all_texts = []
    all_labels = []
    for texts, labels in client_datasets_data:
        all_texts.extend(texts[:20])
        all_labels.extend(labels[:20])
    
    test_dataset = SimpleDataset(all_texts, all_labels, tokenizer, config.max_sequence_length)
    
    print(f"\nInitial global model evaluation:")
    initial_metrics = server.evaluate_global_model(test_dataset)
    print(f"Loss: {initial_metrics['loss']:.4f}, Accuracy: {initial_metrics['accuracy']:.4f}")
    
    # Federated learning rounds
    for round_num in range(config.num_rounds):
        print(f"\n{'='*30} ROUND {round_num + 1} {'='*30}")
        
        # Generate teacher knowledge for this round
        print("Generating teacher knowledge...")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        teacher_knowledge = server.get_teacher_knowledge(test_loader)
        
        # Client training with knowledge distillation
        client_lora_params = []
        client_weights = []
        client_metrics = []
        
        for client in clients:
            print(f"\nTraining KD Client {client.client_id} with teacher knowledge...")
            
            # Local training with knowledge distillation
            metrics = client.local_training_with_distillation(teacher_knowledge)
            client_metrics.append(metrics)
            
            # Get updated LoRA parameters (only from compatible clients)
            updated_lora_params = client.get_lora_parameters()
            client_lora_params.append(updated_lora_params)
            client_weights.append(metrics['num_samples'])
        
        # Server aggregation (only for compatible LoRA parameters)
        print(f"\nAggregating compatible LoRA updates from {len(clients)} clients...")
        aggregated_lora_params = server.aggregate_lora_parameters(client_lora_params, client_weights)
        
        if aggregated_lora_params:
            server.set_lora_parameters(aggregated_lora_params)
            print(f"Updated global model with {len(aggregated_lora_params)} LoRA parameters")
        else:
            print("No compatible LoRA parameters to aggregate (different architectures)")
        
        # Evaluate global model
        global_metrics = server.evaluate_global_model(test_dataset)
        
        # Round summary
        avg_client_loss = np.mean([m['loss'] for m in client_metrics])
        avg_client_accuracy = np.mean([m['accuracy'] for m in client_metrics])
        avg_kd_loss = np.mean([m['kd_loss'] for m in client_metrics])
        
        print(f"\nRound {round_num + 1} Results:")
        print(f"  Average Client Loss: {avg_client_loss:.4f}")
        print(f"  Average Client Accuracy: {avg_client_accuracy:.4f}")
        print(f"  Average KD Loss: {avg_kd_loss:.4f}")
        print(f"  Global Model Loss: {global_metrics['loss']:.4f}")
        print(f"  Global Model Accuracy: {global_metrics['accuracy']:.4f}")
    
    print(f"\n{'='*30} TRAINING COMPLETED {'='*30}")
    print("Knowledge Distillation + LoRA Federated learning completed!")
    print(f"Final Global Model Accuracy: {global_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Distillation + LoRA Federated Learning")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--distillation_alpha", type=float, default=0.7, help="Distillation loss weight")
    
    args = parser.parse_args()
    
    run_knowledge_distillation_federated_learning()
