#!/usr/bin/env python3
"""
Streaming Federated Learning Implementation
Real-time data processing with continuous model updates
"""

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import websockets
import json
import logging
from typing import Dict, List, AsyncGenerator, Optional
import numpy as np
from dataclasses import dataclass, field
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming federated learning"""
    global_model_name: str = "bert-base-uncased"
    client_model_name: str = "prajjwal1/bert-tiny"
    server_host: str = "localhost"
    server_port: int = 8765
    batch_size: int = 8
    learning_rate: float = 2e-5
    buffer_size: int = 100
    update_frequency: int = 10  # Send updates every N batches
    max_sequence_length: int = 128
    num_workers: int = 4


class StreamingDataBuffer:
    """Circular buffer for streaming data"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_sample(self, text: str, label: int):
        """Add a new sample to the buffer"""
        with self.lock:
            self.buffer.append({'text': text, 'label': label, 'timestamp': time.time()})
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get a batch of samples from the buffer"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            # Get most recent samples
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                batch.append(self.buffer.popleft())
            return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


class StreamingDataset(Dataset):
    """Dataset that works with streaming buffer"""
    
    def __init__(self, buffer: StreamingDataBuffer, tokenizer, max_length: int = 128):
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._current_batch = []
        self._refresh_batch()
    
    def _refresh_batch(self):
        """Refresh the current batch from buffer"""
        self._current_batch = self.buffer.get_batch(self.buffer.size())
    
    def __len__(self):
        return len(self._current_batch)
    
    def __getitem__(self, idx):
        if idx >= len(self._current_batch):
            self._refresh_batch()
            if idx >= len(self._current_batch):
                raise IndexError("Index out of range")
        
        sample = self._current_batch[idx]
        text = sample['text']
        label = sample['label']
        
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


class StreamingFederatedServer:
    """Streaming federated learning server"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.clients = {}
        self.global_model = AutoModel.from_pretrained(config.global_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        
        # Global model classifier
        self.classifier = nn.Linear(self.global_model.config.hidden_size, 2)  # Binary classification
        self.classifier.to(self.device)
        
        self.update_queue = asyncio.Queue()
        self.model_version = 0
        
        logger.info(f"Streaming server initialized on {self.device}")
    
    async def handle_client_connection(self, websocket, path):
        """Handle streaming client connections"""
        client_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'register':
                    client_id = data['client_id']
                    self.clients[client_id] = {
                        'websocket': websocket,
                        'last_update': time.time(),
                        'model_version': 0
                    }
                    logger.info(f"Streaming client {client_id} connected")
                    
                    # Send initial model
                    await self.send_model_update(client_id)
                
                elif data['type'] == 'streaming_update':
                    # Handle streaming parameter updates
                    await self.update_queue.put({
                        'client_id': data['client_id'],
                        'parameters': data['parameters'],
                        'batch_metrics': data['metrics'],
                        'timestamp': time.time()
                    })
                
                elif data['type'] == 'data_stream':
                    # Handle incoming data stream
                    logger.info(f"Received {len(data['samples'])} samples from client {data['client_id']}")
        
        except websockets.exceptions.ConnectionClosed:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Streaming client {client_id} disconnected")
    
    async def send_model_update(self, client_id: str):
        """Send model update to specific client"""
        if client_id not in self.clients:
            return
        
        # Extract model parameters
        params = {}
        for name, param in self.global_model.named_parameters():
            params[name] = param.detach().cpu().numpy().tolist()
        
        # Add classifier parameters
        for name, param in self.classifier.named_parameters():
            params[f"classifier.{name}"] = param.detach().cpu().numpy().tolist()
        
        message = {
            'type': 'model_update',
            'parameters': params,
            'version': self.model_version,
            'timestamp': time.time()
        }
        
        try:
            await self.clients[client_id]['websocket'].send(json.dumps(message))
            self.clients[client_id]['model_version'] = self.model_version
        except websockets.exceptions.ConnectionClosed:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def broadcast_model_update(self):
        """Broadcast model update to all clients"""
        for client_id in list(self.clients.keys()):
            await self.send_model_update(client_id)
    
    async def process_streaming_updates(self):
        """Process streaming updates from clients"""
        update_buffer = []
        last_aggregation = time.time()
        aggregation_interval = 5.0  # Aggregate every 5 seconds
        
        while True:
            try:
                # Wait for updates with timeout
                update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                update_buffer.append(update)
                
                # Aggregate if we have enough updates or timeout reached
                current_time = time.time()
                if (len(update_buffer) >= len(self.clients) or 
                    (current_time - last_aggregation) > aggregation_interval):
                    
                    if update_buffer:
                        await self.aggregate_streaming_updates(update_buffer)
                        update_buffer = []
                        last_aggregation = current_time
                
            except asyncio.TimeoutError:
                # Periodic aggregation even without updates
                if update_buffer and (time.time() - last_aggregation) > aggregation_interval:
                    await self.aggregate_streaming_updates(update_buffer)
                    update_buffer = []
                    last_aggregation = time.time()
    
    async def aggregate_streaming_updates(self, updates: List[Dict]):
        """Aggregate streaming updates using weighted averaging"""
        if not updates:
            return
        
        logger.info(f"Aggregating {len(updates)} streaming updates")
        
        # Simple weighted averaging
        aggregated_params = {}
        total_weight = 0
        
        for update in updates:
            weight = update['batch_metrics'].get('batch_size', 1)
            total_weight += weight
            
            for param_name, param_values in update['parameters'].items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_values)
                aggregated_params[param_name] += weight * np.array(param_values)
        
        # Normalize by total weight
        for param_name in aggregated_params:
            aggregated_params[param_name] /= total_weight
        
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.data = torch.tensor(aggregated_params[name]).to(self.device)
            
            for name, param in self.classifier.named_parameters():
                classifier_name = f"classifier.{name}"
                if classifier_name in aggregated_params:
                    param.data = torch.tensor(aggregated_params[classifier_name]).to(self.device)
        
        self.model_version += 1
        
        # Broadcast updated model
        await self.broadcast_model_update()
        
        logger.info(f"Model updated to version {self.model_version}")
    
    async def start_streaming_server(self):
        """Start the streaming federated server"""
        logger.info(f"Starting streaming server on {self.config.server_host}:{self.config.server_port}")
        
        # Start update processor
        asyncio.create_task(self.process_streaming_updates())
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client_connection,
            self.config.server_host,
            self.config.server_port
        )
        
        logger.info("Streaming server ready for connections")
        await server.wait_closed()


class StreamingFederatedClient:
    """Streaming federated learning client"""
    
    def __init__(self, client_id: str, config: StreamingConfig):
        self.client_id = client_id
        self.config = config
        
        # Initialize models
        self.model = AutoModel.from_pretrained(config.client_model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classifier.to(self.device)
        
        # Initialize tokenizer and optimizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model_name)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=config.learning_rate
        )
        
        # Streaming components
        self.data_buffer = StreamingDataBuffer(config.buffer_size)
        self.websocket = None
        self.model_version = 0
        self.batch_count = 0
        
        logger.info(f"Streaming client {client_id} initialized")
    
    async def connect_to_server(self):
        """Connect to streaming server"""
        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        self.websocket = await websockets.connect(uri)
        
        # Register with server
        register_msg = {
            'type': 'register',
            'client_id': self.client_id
        }
        await self.websocket.send(json.dumps(register_msg))
        logger.info(f"Client {self.client_id} connected to streaming server")
    
    async def listen_for_model_updates(self):
        """Listen for model updates from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data['type'] == 'model_update':
                    await self.update_local_model(data['parameters'])
                    self.model_version = data['version']
                    logger.info(f"Client {self.client_id} updated to model version {self.model_version}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {self.client_id} connection closed")
    
    async def update_local_model(self, parameters: Dict):
        """Update local model with global parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data = torch.tensor(parameters[name]).to(self.device)
            
            for name, param in self.classifier.named_parameters():
                classifier_name = f"classifier.{name}"
                if classifier_name in parameters:
                    param.data = torch.tensor(parameters[classifier_name]).to(self.device)
    
    async def process_streaming_data(self):
        """Process streaming data continuously"""
        while True:
            if self.data_buffer.size() >= self.config.batch_size:
                await self.train_on_batch()
            else:
                await asyncio.sleep(0.1)  # Wait for more data
    
    async def train_on_batch(self):
        """Train on a batch of streaming data"""
        # Get batch from buffer
        batch_data = self.data_buffer.get_batch(self.config.batch_size)
        if not batch_data:
            return
        
        # Prepare batch
        texts = [item['text'] for item in batch_data]
        labels = [item['label'] for item in batch_data]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        labels = torch.tensor(labels).to(self.device)
        
        # Forward pass
        self.model.train()
        self.classifier.train()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        
        self.batch_count += 1
        
        logger.info(f"Client {self.client_id} batch {self.batch_count} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        # Send update to server periodically
        if self.batch_count % self.config.update_frequency == 0:
            await self.send_parameter_update(loss.item(), accuracy, len(batch_data))
    
    async def send_parameter_update(self, loss: float, accuracy: float, batch_size: int):
        """Send parameter update to server"""
        # Extract model parameters
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().cpu().numpy().tolist()
        
        for name, param in self.classifier.named_parameters():
            params[f"classifier.{name}"] = param.detach().cpu().numpy().tolist()
        
        update_msg = {
            'type': 'streaming_update',
            'client_id': self.client_id,
            'parameters': params,
            'metrics': {
                'loss': loss,
                'accuracy': accuracy,
                'batch_size': batch_size,
                'batch_count': self.batch_count
            }
        }
        
        await self.websocket.send(json.dumps(update_msg))
        logger.info(f"Client {self.client_id} sent parameter update (batch {self.batch_count})")
    
    def add_streaming_data(self, text: str, label: int):
        """Add new data to the streaming buffer"""
        self.data_buffer.add_sample(text, label)
    
    async def simulate_data_stream(self):
        """Simulate incoming data stream"""
        import random
        
        sample_texts = [
            "This is a positive example",
            "This is a negative example",
            "Great product, highly recommend",
            "Terrible service, very disappointed",
            "Amazing experience, will come back",
            "Poor quality, waste of money"
        ]
        
        while True:
            # Simulate incoming data
            text = random.choice(sample_texts) + f" (stream {time.time()})"
            label = random.randint(0, 1)
            
            self.add_streaming_data(text, label)
            
            # Random delay between data arrivals
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def run_streaming_client(self):
        """Run the streaming federated client"""
        await self.connect_to_server()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self.listen_for_model_updates()),
            asyncio.create_task(self.process_streaming_data()),
            asyncio.create_task(self.simulate_data_stream())
        ]
        
        await asyncio.gather(*tasks)


# Example usage functions
async def run_streaming_server():
    """Run streaming federated server"""
    config = StreamingConfig()
    server = StreamingFederatedServer(config)
    await server.start_streaming_server()


async def run_streaming_client(client_id: str):
    """Run streaming federated client"""
    config = StreamingConfig()
    client = StreamingFederatedClient(client_id, config)
    await client.run_streaming_client()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            asyncio.run(run_streaming_server())
        elif sys.argv[1] == "client":
            client_id = sys.argv[2] if len(sys.argv) > 2 else "client_0"
            asyncio.run(run_streaming_client(client_id))
        else:
            print("Usage: python streaming_federated.py [server|client] [client_id]")
    else:
        print("Streaming Federated Learning System")
        print("Usage: python streaming_federated.py [server|client] [client_id]")
