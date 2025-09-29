#!/usr/bin/env python3
"""
Custom Federated Learning Implementation
- Global Model: BERT (Server)
- Client Models: Tiny-BERT
- Full Streaming Support
- No Flower Framework
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertConfig,
    get_linear_schedule_with_warmup
)
import asyncio
import websockets
import json
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated learning setup"""
    global_model_name: str = "bert-base-uncased"
    client_model_name: str = "prajjwal1/bert-tiny"
    num_clients: int = 3
    server_port: int = 8765
    client_ports: List[int] = None
    learning_rate: float = 2e-5
    local_epochs: int = 3
    global_rounds: int = 10
    batch_size: int = 16
    max_sequence_length: int = 128
    knowledge_distillation_alpha: float = 0.7
    temperature: float = 4.0
    
    def __post_init__(self):
        if self.client_ports is None:
            self.client_ports = [8766 + i for i in range(self.num_clients)]


class GlobalBERTModel(nn.Module):
    """Global BERT model on server"""
    
    def __init__(self, config: FederatedConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.global_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
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
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if return_hidden_states else None,
            'pooler_output': outputs.pooler_output
        }


class TinyBERTClient(nn.Module):
    """Tiny-BERT model for clients"""
    
    def __init__(self, config: FederatedConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.client_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        
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
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if return_hidden_states else None,
            'pooler_output': outputs.pooler_output
        }


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for teacher-student learning"""
    
    def __init__(self, alpha: float = 0.7, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Task loss
        task_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        return total_loss, distillation_loss, task_loss


class FederatedServer:
    """Central server for federated learning"""
    
    def __init__(self, config: FederatedConfig, num_labels: int = 2):
        self.config = config
        self.global_model = GlobalBERTModel(config, num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        self.clients = {}
        self.client_updates = queue.Queue()
        self.current_round = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config.global_model_name)
        
        logger.info(f"Server initialized with global model: {config.global_model_name}")
        logger.info(f"Device: {self.device}")
    
    async def handle_client(self, websocket, path):
        """Handle client connections and communications"""
        try:
            client_id = None
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'register':
                    client_id = data['client_id']
                    self.clients[client_id] = websocket
                    logger.info(f"Client {client_id} registered")
                    
                    # Send initial global model
                    await self.send_global_model(client_id)
                    
                elif data['type'] == 'update':
                    # Receive client update
                    client_update = {
                        'client_id': data['client_id'],
                        'parameters': data['parameters'],
                        'metrics': data['metrics'],
                        'round': data['round']
                    }
                    self.client_updates.put(client_update)
                    logger.info(f"Received update from client {data['client_id']} for round {data['round']}")
                    
        except websockets.exceptions.ConnectionClosed:
            if client_id:
                logger.info(f"Client {client_id} disconnected")
                if client_id in self.clients:
                    del self.clients[client_id]
    
    async def send_global_model(self, client_id):
        """Send global model parameters to client"""
        if client_id not in self.clients:
            return
            
        # Extract model parameters
        global_params = {}
        for name, param in self.global_model.named_parameters():
            global_params[name] = param.detach().cpu().numpy().tolist()
        
        message = {
            'type': 'global_model',
            'parameters': global_params,
            'round': self.current_round
        }
        
        try:
            await self.clients[client_id].send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Failed to send to client {client_id} - connection closed")
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def broadcast_global_model(self):
        """Broadcast updated global model to all clients"""
        for client_id in list(self.clients.keys()):
            await self.send_global_model(client_id)
    
    def aggregate_updates(self, client_updates: List[Dict]):
        """Aggregate client updates using FedAvg"""
        logger.info(f"Aggregating updates from {len(client_updates)} clients")
        
        # Simple FedAvg implementation
        aggregated_params = {}
        total_samples = sum(update['metrics']['num_samples'] for update in client_updates)
        
        for update in client_updates:
            weight = update['metrics']['num_samples'] / total_samples
            params = update['parameters']
            
            for param_name, param_values in params.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_values)
                aggregated_params[param_name] += weight * np.array(param_values)
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.data = torch.tensor(aggregated_params[name]).to(self.device)
        
        logger.info("Global model updated with aggregated parameters")
    
    async def run_federated_rounds(self):
        """Run federated learning rounds"""
        for round_num in range(self.config.global_rounds):
            self.current_round = round_num
            logger.info(f"Starting federated round {round_num + 1}/{self.config.global_rounds}")
            
            # Wait for client updates
            updates = []
            expected_clients = len(self.clients)
            
            # Wait for updates from all clients (with timeout)
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            
            while len(updates) < expected_clients and (time.time() - start_time) < timeout:
                try:
                    update = self.client_updates.get(timeout=10)
                    if update['round'] == round_num:
                        updates.append(update)
                except queue.Empty:
                    continue
            
            if updates:
                # Aggregate updates
                self.aggregate_updates(updates)
                
                # Broadcast updated global model
                await self.broadcast_global_model()
                
                # Log round metrics
                avg_loss = np.mean([u['metrics']['loss'] for u in updates])
                avg_accuracy = np.mean([u['metrics']['accuracy'] for u in updates])
                logger.info(f"Round {round_num + 1} completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
            else:
                logger.warning(f"No updates received for round {round_num + 1}")
            
            await asyncio.sleep(1)  # Brief pause between rounds
    
    async def start_server(self):
        """Start the federated learning server"""
        logger.info(f"Starting federated server on port {self.config.server_port}")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            "localhost",
            self.config.server_port
        )
        
        logger.info("Server started, waiting for clients...")
        
        # Wait for clients to connect
        await asyncio.sleep(5)
        
        # Start federated learning rounds
        await self.run_federated_rounds()
        
        # Keep server running
        await server.wait_closed()


class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: int, config: FederatedConfig, dataset, num_labels: int = 2):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.model = TinyBERTClient(config, num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.kd_loss = KnowledgeDistillationLoss(config.knowledge_distillation_alpha, config.temperature)
        
        self.websocket = None
        self.global_model_params = None
        self.current_round = 0
        
        logger.info(f"Client {client_id} initialized with {len(dataset)} samples")
    
    async def connect_to_server(self):
        """Connect to the federated server"""
        uri = f"ws://localhost:{self.config.server_port}"
        try:
            self.websocket = await websockets.connect(uri)
            
            # Register with server
            register_msg = {
                'type': 'register',
                'client_id': self.client_id
            }
            await self.websocket.send(json.dumps(register_msg))
            logger.info(f"Client {self.client_id} connected to server")
            
        except Exception as e:
            logger.error(f"Client {self.client_id} failed to connect: {e}")
            raise
    
    async def listen_for_updates(self):
        """Listen for global model updates from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data['type'] == 'global_model':
                    self.global_model_params = data['parameters']
                    self.current_round = data['round']
                    logger.info(f"Client {self.client_id} received global model for round {self.current_round}")
                    
                    # Start local training
                    asyncio.create_task(self.local_training())
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {self.client_id} connection closed")
    
    async def local_training(self):
        """Perform local training with knowledge distillation"""
        logger.info(f"Client {self.client_id} starting local training for round {self.current_round}")
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student (client) forward pass
                student_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # If we have global model params, use knowledge distillation
                if self.global_model_params:
                    # Create teacher logits (simplified - in practice, you'd load global model)
                    teacher_logits = self.get_teacher_logits(input_ids, attention_mask)
                    loss, kd_loss, task_loss = self.kd_loss(
                        student_outputs['logits'],
                        teacher_logits,
                        labels
                    )
                else:
                    loss = student_outputs['loss']
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(student_outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        logger.info(f"Client {self.client_id} local training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Send update to server
        await self.send_update_to_server(avg_loss, accuracy, total_samples)
    
    def get_teacher_logits(self, input_ids, attention_mask):
        """Get teacher (global model) logits for knowledge distillation"""
        # Simplified implementation - in practice, you'd load the global model
        # For now, return dummy logits
        batch_size = input_ids.size(0)
        num_labels = 2  # Assuming binary classification
        return torch.randn(batch_size, num_labels).to(self.device)
    
    async def send_update_to_server(self, loss, accuracy, num_samples):
        """Send local model update to server"""
        # Extract model parameters
        local_params = {}
        for name, param in self.model.named_parameters():
            local_params[name] = param.detach().cpu().numpy().tolist()
        
        update_msg = {
            'type': 'update',
            'client_id': self.client_id,
            'parameters': local_params,
            'metrics': {
                'loss': loss,
                'accuracy': accuracy,
                'num_samples': num_samples
            },
            'round': self.current_round
        }
        
        await self.websocket.send(json.dumps(update_msg))
        logger.info(f"Client {self.client_id} sent update to server")
    
    async def run_client(self):
        """Run the federated client"""
        await self.connect_to_server()
        await self.listen_for_updates()


# Example dataset class
class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration"""
    
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


async def run_server(config: FederatedConfig):
    """Run the federated server"""
    server = FederatedServer(config)
    await server.start_server()


async def run_client(client_id: int, config: FederatedConfig, dataset):
    """Run a federated client"""
    client = FederatedClient(client_id, config, dataset)
    await client.run_client()


def create_dummy_datasets(num_clients: int = 3):
    """Create dummy datasets for testing"""
    datasets = []
    
    for i in range(num_clients):
        # Create different data distributions for each client
        texts = [f"This is sample text {j} for client {i}" for j in range(100)]
        labels = [j % 2 for j in range(100)]  # Binary labels
        datasets.append((texts, labels))
    
    return datasets


async def main():
    """Main function to run server or client based on command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom Federated Learning System")
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                        help="Run as server or client")
    parser.add_argument("--client_id", type=int, default=0,
                        help="Client ID (only used in client mode)")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Total number of clients")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of federated learning rounds")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="Number of local training epochs per round")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FederatedConfig(
        num_clients=args.num_clients,
        global_rounds=args.num_rounds,
        local_epochs=args.local_epochs
    )
    
    print("Custom Federated Learning System Initialized")
    print(f"Mode: {args.mode}")
    print(f"Global Model: {config.global_model_name}")
    print(f"Client Model: {config.client_model_name}")
    print(f"Number of Clients: {config.num_clients}")
    print(f"Global Rounds: {config.global_rounds}")
    print("-" * 50)
    
    if args.mode == "server":
        print("Starting Federated Server...")
        server = FederatedServer(config)
        await server.start_server()
        
    elif args.mode == "client":
        print(f"Starting Client {args.client_id}...")
        
        # Create dummy dataset for this client
        dummy_data = create_dummy_datasets(config.num_clients)
        client_texts, client_labels = dummy_data[args.client_id]
        
        # Create dataset
        tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
        dataset = SimpleTextDataset(client_texts, client_labels, tokenizer)
        
        # Start client
        client = FederatedClient(args.client_id, config, dataset)
        await client.run_client()


if __name__ == "__main__":
    asyncio.run(main())
