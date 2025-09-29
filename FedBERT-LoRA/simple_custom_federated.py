#!/usr/bin/env python3
"""
Simplified Custom Federated Learning Demo
- Global Model: BERT 
- Client Models: Tiny-BERT
- Sequential training for demonstration
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleFedConfig:
    """Simple configuration for federated learning"""
    global_model_name: str = "bert-base-uncased"
    client_model_name: str = "prajjwal1/bert-tiny"
    num_clients: int = 3
    num_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_sequence_length: int = 128


class SimpleGlobalModel(nn.Module):
    """Simplified Global BERT model"""
    
    def __init__(self, config: SimpleFedConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.global_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return {'loss': loss, 'logits': logits}


class SimpleClientModel(nn.Module):
    """Simplified Client Tiny-BERT model"""
    
    def __init__(self, config: SimpleFedConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.client_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return {'loss': loss, 'logits': logits}


class SimpleDataset(Dataset):
    """Simple text dataset"""
    
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


class SimpleFederatedServer:
    """Simplified Federated Server"""
    
    def __init__(self, config: SimpleFedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = SimpleGlobalModel(config)
        self.global_model.to(self.device)
        
        logger.info(f"Server initialized with global model: {config.global_model_name}")
        logger.info(f"Device: {self.device}")
    
    def get_global_parameters(self) -> Dict[str, np.ndarray]:
        """Get global model parameters"""
        params = {}
        for name, param in self.global_model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params
    
    def set_global_parameters(self, params: Dict[str, np.ndarray]):
        """Set global model parameters"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in params:
                    param.data = torch.tensor(params[name]).to(self.device)
    
    def aggregate_parameters(self, client_params: List[Dict[str, np.ndarray]], 
                           client_weights: List[float]) -> Dict[str, np.ndarray]:
        """Aggregate client parameters using weighted averaging (only classifier layers)"""
        aggregated_params = {}
        total_weight = sum(client_weights)
        
        # Get global model parameter shapes for comparison
        global_param_shapes = {}
        for name, param in self.global_model.named_parameters():
            global_param_shapes[name] = param.shape
        
        # Initialize aggregated parameters (only for compatible layers)
        for name in client_params[0].keys():
            client_shape = client_params[0][name].shape
            if name in global_param_shapes and global_param_shapes[name] == client_shape:
                aggregated_params[name] = np.zeros_like(client_params[0][name])
        
        # Weighted averaging (only for compatible parameters)
        for client_param, weight in zip(client_params, client_weights):
            normalized_weight = weight / total_weight
            for name, param in client_param.items():
                if name in aggregated_params:
                    aggregated_params[name] += normalized_weight * param
        
        logger.info(f"Aggregated {len(aggregated_params)} compatible parameter groups")
        return aggregated_params
    
    def evaluate_global_model(self, test_dataset) -> Dict[str, float]:
        """Evaluate global model on test data"""
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


class SimpleFederatedClient:
    """Simplified Federated Client"""
    
    def __init__(self, client_id: int, config: SimpleFedConfig, dataset: SimpleDataset):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SimpleClientModel(config)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        logger.info(f"Client {client_id} initialized with {len(dataset)} samples")
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set client model parameters (only compatible layers)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    global_param = torch.tensor(params[name]).to(self.device)
                    # Only update if shapes match (skip BERT layers, update classifier only)
                    if param.shape == global_param.shape:
                        param.data = global_param
                    else:
                        logger.info(f"Skipping parameter {name} due to shape mismatch: {param.shape} vs {global_param.shape}")
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get client model parameters"""
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params
    
    def local_training(self) -> Dict[str, float]:
        """Perform local training"""
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
        
        logger.info(f"Client {self.client_id} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'num_samples': len(self.dataset)}


def create_dummy_datasets(num_clients: int = 3, samples_per_client: int = 100):
    """Create dummy datasets for each client"""
    datasets = []
    
    # Different types of texts for each client to simulate heterogeneous data
    base_texts = [
        ["This is a positive sentiment", "Great product, love it", "Amazing experience", "Wonderful service"],
        ["This is negative feedback", "Terrible quality", "Very disappointed", "Poor customer service"],
        ["Neutral statement here", "Average product quality", "Standard service", "Regular experience"]
    ]
    
    for i in range(num_clients):
        texts = []
        labels = []
        
        # Create diverse data for each client
        base_text_set = base_texts[i % len(base_texts)]
        
        for j in range(samples_per_client):
            # Add variety to the texts
            text = f"{base_text_set[j % len(base_text_set)]} - sample {j} from client {i}"
            label = j % 2  # Binary classification
            
            texts.append(text)
            labels.append(label)
        
        datasets.append((texts, labels))
        logger.info(f"Created dataset for client {i} with {len(texts)} samples")
    
    return datasets


def run_simple_federated_learning():
    """Run the simplified federated learning demonstration"""
    config = SimpleFedConfig(num_clients=3, num_rounds=3, local_epochs=2)
    
    print("=" * 60)
    print("SIMPLIFIED CUSTOM FEDERATED LEARNING DEMO")
    print("=" * 60)
    print(f"Global Model: {config.global_model_name}")
    print(f"Client Model: {config.client_model_name}")
    print(f"Number of Clients: {config.num_clients}")
    print(f"Number of Rounds: {config.num_rounds}")
    print(f"Local Epochs per Round: {config.local_epochs}")
    print("=" * 60)
    
    # Initialize server
    server = SimpleFederatedServer(config)
    
    # Create client datasets
    client_datasets_data = create_dummy_datasets(config.num_clients)
    
    # Initialize clients
    clients = []
    tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
    
    for i in range(config.num_clients):
        texts, labels = client_datasets_data[i]
        dataset = SimpleDataset(texts, labels, tokenizer, config.max_sequence_length)
        client = SimpleFederatedClient(i, config, dataset)
        clients.append(client)
    
    # Create test dataset (combination of all client data)
    all_texts = []
    all_labels = []
    for texts, labels in client_datasets_data:
        all_texts.extend(texts[:20])  # Take first 20 samples from each client for testing
        all_labels.extend(labels[:20])
    
    test_dataset = SimpleDataset(all_texts, all_labels, tokenizer, config.max_sequence_length)
    
    print(f"\nInitial global model evaluation:")
    initial_metrics = server.evaluate_global_model(test_dataset)
    print(f"Loss: {initial_metrics['loss']:.4f}, Accuracy: {initial_metrics['accuracy']:.4f}")
    
    # Federated learning rounds
    for round_num in range(config.num_rounds):
        print(f"\n{'='*20} ROUND {round_num + 1} {'='*20}")
        
        # Get current global parameters
        global_params = server.get_global_parameters()
        
        # Client training
        client_params = []
        client_weights = []
        client_metrics = []
        
        for client in clients:
            print(f"\nTraining Client {client.client_id}...")
            
            # Set client model to current global parameters
            client.set_parameters(global_params)
            
            # Local training
            metrics = client.local_training()
            client_metrics.append(metrics)
            
            # Get updated parameters
            updated_params = client.get_parameters()
            client_params.append(updated_params)
            client_weights.append(metrics['num_samples'])
        
        # Server aggregation
        print(f"\nAggregating updates from {len(clients)} clients...")
        aggregated_params = server.aggregate_parameters(client_params, client_weights)
        server.set_global_parameters(aggregated_params)
        
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
    
    print(f"\n{'='*20} TRAINING COMPLETED {'='*20}")
    print("Federated learning demonstration completed successfully!")
    print(f"Final Global Model Accuracy: {global_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Custom Federated Learning Demo")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = SimpleFedConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs
    )
    
    run_simple_federated_learning()
