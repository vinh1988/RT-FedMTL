#!/usr/bin/env python3
"""
LoRA-based Federated Learning
- Solves incompatible layer problem using Low-Rank Adaptation
- Global BERT + Client Tiny-BERT can share LoRA parameters
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoRAFedConfig:
    """Configuration for LoRA-based federated learning"""
    global_model_name: str = "bert-base-uncased"
    client_model_name: str = "prajjwal1/bert-tiny"
    num_clients: int = 3
    num_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_sequence_length: int = 128
    
    # LoRA parameters
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling parameter
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value", "dense"]


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""
    
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA_A with random values, LoRA_B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # LoRA forward: x @ (A @ B) * scaling
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return lora_out


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, original_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features, 
            original_layer.out_features, 
            r, alpha, dropout
        )
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original output + LoRA adaptation
        original_out = self.original_layer(x)
        lora_out = self.lora(x)
        return original_out + lora_out


def add_lora_to_model(model, config: LoRAFedConfig):
    """Add LoRA adapters to target modules in the model"""
    lora_modules = {}
    
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                lora_linear = LoRALinear(
                    module, 
                    config.lora_r, 
                    config.lora_alpha, 
                    config.lora_dropout
                )
                
                # Set the LoRA module in the model
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


def get_lora_parameters(model) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model"""
    lora_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params[f"{name}.lora_A"] = module.lora.lora_A.data
            lora_params[f"{name}.lora_B"] = module.lora.lora_B.data
    
    return lora_params


def set_lora_parameters(model, lora_params: Dict[str, torch.Tensor]):
    """Set LoRA parameters in model"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_name = f"{name}.lora_A"
            lora_b_name = f"{name}.lora_B"
            
            if lora_a_name in lora_params:
                module.lora.lora_A.data = lora_params[lora_a_name].clone()
            if lora_b_name in lora_params:
                module.lora.lora_B.data = lora_params[lora_b_name].clone()


class LoRAFederatedModel(nn.Module):
    """Model with LoRA adapters for federated learning"""
    
    def __init__(self, model_name: str, config: LoRAFedConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        # Add LoRA adapters
        self.lora_modules = add_lora_to_model(self.bert, config)
        
        logger.info(f"Model initialized: {model_name}")
        logger.info(f"Added LoRA to {len(self.lora_modules)} modules")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


class LoRAFederatedServer:
    """Federated server that aggregates LoRA parameters"""
    
    def __init__(self, config: LoRAFedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Global model with LoRA
        self.global_model = LoRAFederatedModel(config.global_model_name, config)
        self.global_model.to(self.device)
        
        logger.info(f"LoRA Server initialized: {config.global_model_name}")
        logger.info(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    
    def get_global_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Get global LoRA parameters"""
        lora_params = get_lora_parameters(self.global_model)
        return {name: param.detach().cpu().numpy() for name, param in lora_params.items()}
    
    def set_global_lora_parameters(self, lora_params: Dict[str, np.ndarray]):
        """Set global LoRA parameters"""
        tensor_params = {name: torch.tensor(param).to(self.device) for name, param in lora_params.items()}
        set_lora_parameters(self.global_model, tensor_params)
    
    def aggregate_lora_parameters(self, client_lora_params: List[Dict[str, np.ndarray]], 
                                 client_weights: List[float]) -> Dict[str, np.ndarray]:
        """Aggregate LoRA parameters from clients"""
        aggregated_params = {}
        total_weight = sum(client_weights)
        
        # Get all LoRA parameter names
        all_param_names = set()
        for client_params in client_lora_params:
            all_param_names.update(client_params.keys())
        
        # Aggregate each LoRA parameter
        for param_name in all_param_names:
            # Check if all clients have this parameter
            if all(param_name in client_params for client_params in client_lora_params):
                # Initialize with zeros
                param_shape = client_lora_params[0][param_name].shape
                aggregated_params[param_name] = np.zeros(param_shape)
                
                # Weighted average
                for client_params, weight in zip(client_lora_params, client_weights):
                    normalized_weight = weight / total_weight
                    aggregated_params[param_name] += normalized_weight * client_params[param_name]
        
        logger.info(f"Aggregated {len(aggregated_params)} LoRA parameters")
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


class LoRAFederatedClient:
    """Federated client with LoRA adapters"""
    
    def __init__(self, client_id: int, config: LoRAFedConfig, dataset):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Client model with LoRA
        self.model = LoRAFederatedModel(config.client_model_name, config)
        self.model.to(self.device)
        
        # Only train LoRA parameters + classifier
        self.optimizer = torch.optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if 'lora' in n or 'classifier' in n]},
        ], lr=config.learning_rate)
        
        logger.info(f"LoRA Client {client_id} initialized with {len(dataset)} samples")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def set_lora_parameters(self, lora_params: Dict[str, np.ndarray]):
        """Set LoRA parameters from server"""
        tensor_params = {name: torch.tensor(param).to(self.device) for name, param in lora_params.items()}
        set_lora_parameters(self.model, tensor_params)
    
    def get_lora_parameters(self) -> Dict[str, np.ndarray]:
        """Get client's LoRA parameters"""
        lora_params = get_lora_parameters(self.model)
        return {name: param.detach().cpu().numpy() for name, param in lora_params.items()}
    
    def local_training(self) -> Dict[str, float]:
        """Perform local training with LoRA"""
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / (len(dataloader) * self.config.local_epochs)
        accuracy = correct_predictions / total_samples
        
        logger.info(f"LoRA Client {self.client_id} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'num_samples': len(self.dataset)}


# Simple dataset class (reuse from previous implementation)
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


def run_lora_federated_learning():
    """Run LoRA-based federated learning"""
    config = LoRAFedConfig(num_clients=3, num_rounds=3, local_epochs=2, lora_r=16)
    
    print("=" * 70)
    print("LoRA-BASED FEDERATED LEARNING DEMO")
    print("=" * 70)
    print(f"Global Model: {config.global_model_name}")
    print(f"Client Model: {config.client_model_name}")
    print(f"LoRA Rank: {config.lora_r}, Alpha: {config.lora_alpha}")
    print(f"Number of Clients: {config.num_clients}")
    print(f"Number of Rounds: {config.num_rounds}")
    print("=" * 70)
    
    # Initialize server
    server = LoRAFederatedServer(config)
    
    # Create client datasets
    client_datasets_data = create_dummy_datasets(config.num_clients)
    
    # Initialize clients
    clients = []
    tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
    
    for i in range(config.num_clients):
        texts, labels = client_datasets_data[i]
        dataset = SimpleDataset(texts, labels, tokenizer, config.max_sequence_length)
        client = LoRAFederatedClient(i, config, dataset)
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
        print(f"\n{'='*25} ROUND {round_num + 1} {'='*25}")
        
        # Get current global LoRA parameters
        global_lora_params = server.get_global_lora_parameters()
        
        # Client training
        client_lora_params = []
        client_weights = []
        client_metrics = []
        
        for client in clients:
            print(f"\nTraining LoRA Client {client.client_id}...")
            
            # Set client LoRA parameters to global parameters
            client.set_lora_parameters(global_lora_params)
            
            # Local training
            metrics = client.local_training()
            client_metrics.append(metrics)
            
            # Get updated LoRA parameters
            updated_lora_params = client.get_lora_parameters()
            client_lora_params.append(updated_lora_params)
            client_weights.append(metrics['num_samples'])
        
        # Server aggregation
        print(f"\nAggregating LoRA updates from {len(clients)} clients...")
        aggregated_lora_params = server.aggregate_lora_parameters(client_lora_params, client_weights)
        server.set_global_lora_parameters(aggregated_lora_params)
        
        # Evaluate global model
        global_metrics = server.evaluate_global_model(test_dataset)
        
        # Round summary
        avg_client_loss = np.mean([m['loss'] for m in client_metrics])
        avg_client_accuracy = np.mean([m['accuracy'] for m in client_metrics])
        
        print(f"\nRound {round_num + 1} Results:")
        print(f"  Average Client Loss: {avg_client_loss:.4f}")
        print(f"  Average Client Accuracy: {avg_client_accuracy:.4f}")
        print(f"  Global Model Loss: {global_metrics['loss']:.4f}")
        print(f"  Global Model Accuracy: {global_metrics['accuracy']:.4f}")
    
    print(f"\n{'='*25} TRAINING COMPLETED {'='*25}")
    print("LoRA Federated learning demonstration completed successfully!")
    print(f"Final Global Model Accuracy: {global_metrics['accuracy']:.4f}")
    
    # Show parameter efficiency
    total_params = sum(p.numel() for p in server.global_model.parameters())
    lora_params = sum(p.numel() for n, p in server.global_model.named_parameters() if 'lora' in n)
    print(f"\nParameter Efficiency:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  LoRA Parameters: {lora_params:,}")
    print(f"  LoRA Ratio: {100*lora_params/total_params:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA-based Federated Learning Demo")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = LoRAFedConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    run_lora_federated_learning()
