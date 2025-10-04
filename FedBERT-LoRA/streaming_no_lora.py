#!/usr/bin/env python3
"""
Streaming Federated Learning WITHOUT LoRA
Pure Knowledge Distillation + Cross-Architecture Learning

Features:
- No parameter-efficient fine-tuning (full model training)
- Pure knowledge distillation for cross-architecture transfer
- Real-time WebSocket streaming
- Multi-task learning (SST-2, QQP, STS-B)
- Heterogeneous client models (Tiny-BERT) with server model (BERT-base)

Usage:
    # Start server
    python3 streaming_no_lora.py --mode server --port 8768 --rounds 3
    
    # Start clients (in separate terminals)
    python3 streaming_no_lora.py --mode client --client_id client_sst2 --task sst2 --port 8768
    python3 streaming_no_lora.py --mode client --client_id client_qqp --task qqp --port 8768
    python3 streaming_no_lora.py --mode client --client_id client_stsb --task stsb --port 8768
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import argparse
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NoLoRAConfig:
    """Configuration for streaming federated learning WITHOUT LoRA"""
    # Model configurations
    server_model: str = "bert-base-uncased"
    client_model: str = "prajjwal1/bert-tiny"
    
    # Training parameters
    num_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_samples_per_client: int = 100
    
    # Knowledge distillation parameters
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7  # Weight for KD loss vs task loss
    
    # Server parameters
    server_host: str = "localhost"
    server_port: int = 8768
    max_clients: int = 3
    client_timeout: int = 300  # 5 minutes

class GLUEDataset(Dataset):
    """Dataset for GLUE tasks without LoRA preprocessing"""
    
    def __init__(self, texts: List[str], labels: List[Any], tokenizer, task: str, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        if isinstance(text, list) and len(text) == 2:  # Sentence pairs (QQP)
            encoding = self.tokenizer(
                text[0], text[1],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:  # Single sentences (SST-2, STS-B)
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        # Process labels based on task
        if self.task == 'stsb':
            # Regression task - normalize to 0-1 range
            label = float(label) / 5.0
        else:
            # Classification tasks
            label = int(label)
            
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32 if self.task == 'stsb' else torch.long)
        }

class NoLoRABERTModel(nn.Module):
    """BERT model without LoRA - full parameter training"""
    
    def __init__(self, model_name: str, num_labels: int, task_type: str = 'classification'):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        
        # Load pre-trained model
        if task_type == 'classification':
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels
            )
        else:  # regression
            self.bert = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
            
        logger.info(f"Initialized {model_name} for {task_type} with {num_labels} labels")
        
    def forward(self, input_ids, attention_mask, labels=None):
        if self.task_type == 'classification':
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs
        else:  # regression
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(), labels)
                
            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits,
                'hidden_states': outputs.last_hidden_state
            })()
    
    def get_trainable_parameters(self):
        """Get all trainable parameters (no LoRA filtering)"""
        return {name: param for name, param in self.named_parameters() if param.requires_grad}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters (full model update)"""
        model_dict = self.state_dict()
        
        # Update only compatible parameters
        updated_params = {}
        for name, param in parameters.items():
            if name in model_dict and model_dict[name].shape == param.shape:
                updated_params[name] = param
                
        model_dict.update(updated_params)
        self.load_state_dict(model_dict, strict=False)
        
        logger.info(f"Updated {len(updated_params)}/{len(parameters)} parameters")

def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, task_type):
    """Compute knowledge distillation loss without LoRA complications"""
    
    # Handle dimension mismatches
    if student_logits.dim() != teacher_logits.dim():
        if student_logits.dim() == 1:
            student_logits = student_logits.unsqueeze(0)
        if teacher_logits.dim() == 1:
            teacher_logits = teacher_logits.unsqueeze(0)
    
    # Ensure same batch size
    min_batch_size = min(student_logits.size(0), teacher_logits.size(0))
    student_logits = student_logits[:min_batch_size]
    teacher_logits = teacher_logits[:min_batch_size]
    labels = labels[:min_batch_size]
    
    if task_type == 'regression':
        # For regression: direct MSE between outputs
        kd_loss = nn.MSELoss()(student_logits.squeeze(), teacher_logits.squeeze())
        task_loss = nn.MSELoss()(student_logits.squeeze(), labels)
    else:
        # For classification: standard KD with temperature scaling
        if student_logits.size(-1) != teacher_logits.size(-1):
            # Handle different output dimensions
            min_dim = min(student_logits.size(-1), teacher_logits.size(-1))
            student_logits = student_logits[:, :min_dim]
            teacher_logits = teacher_logits[:, :min_dim]
        
        # Soft targets from teacher
        soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / temperature, dim=-1)
        
        kd_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (temperature ** 2)
        task_loss = nn.CrossEntropyLoss()(student_logits, labels.long())
    
    # Combine losses
    total_loss = alpha * kd_loss + (1 - alpha) * task_loss
    
    return total_loss, kd_loss, task_loss

def load_glue_data(task: str, max_samples: int = 100) -> Tuple[List, List]:
    """Load GLUE dataset for specific task"""
    try:
        if task.lower() == 'sst2':
            dataset = load_dataset('sst2', split='train')
            texts = [item['sentence'] for item in dataset.select(range(min(max_samples, len(dataset))))]
            labels = [item['label'] for item in dataset.select(range(min(max_samples, len(dataset))))]
            logger.info(f"Loaded {len(texts)} samples from SST2")
            
        elif task.lower() == 'qqp':
            dataset = load_dataset('qqp', split='train')
            texts = [[item['question1'], item['question2']] for item in dataset.select(range(min(max_samples, len(dataset))))]
            labels = [item['label'] for item in dataset.select(range(min(max_samples, len(dataset))))]
            logger.info(f"Loaded {len(texts)} samples from QQP")
            
        elif task.lower() == 'stsb':
            dataset = load_dataset('stsb', split='train')
            texts = [[item['sentence1'], item['sentence2']] for item in dataset.select(range(min(max_samples, len(dataset))))]
            labels = [item['label'] for item in dataset.select(range(min(max_samples, len(dataset))))]
            logger.info(f"Loaded {len(texts)} samples from STS-B")
            
        else:
            raise ValueError(f"Unsupported task: {task}")
            
        return texts, labels
        
    except Exception as e:
        logger.warning(f"Failed to load {task} dataset: {e}")
        logger.info("Using dummy data for demonstration")
        
        # Fallback dummy data
        if task.lower() == 'sst2':
            texts = ["This is great!", "This is terrible!"] * (max_samples // 2)
            labels = [1, 0] * (max_samples // 2)
        elif task.lower() == 'qqp':
            texts = [["How are you?", "How do you do?"], ["What is AI?", "What is ML?"]] * (max_samples // 2)
            labels = [1, 0] * (max_samples // 2)
        else:  # stsb
            texts = [["I love this", "I like this"], ["Good morning", "Hello there"]] * (max_samples // 2)
            labels = [4.0, 2.5] * (max_samples // 2)
            
        return texts, labels

class NoLoRAFederatedServer:
    """Federated server without LoRA - manages pure knowledge distillation"""
    
    def __init__(self, config: NoLoRAConfig):
        self.config = config
        self.clients = {}
        self.client_models = {}
        self.current_round = 0
        
        # Initialize server model (teacher)
        self.tokenizer = AutoTokenizer.from_pretrained(config.server_model)
        
        # Server model for multi-task learning (we'll use classification setup)
        self.server_model = NoLoRABERTModel(
            config.server_model, 
            num_labels=2,  # Binary classification as default
            task_type='classification'
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.server_model.parameters())
        trainable_params = sum(p.numel() for p in self.server_model.parameters() if p.requires_grad)
        
        logger.info(f"Server initialized with {config.server_model}")
        logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")
        
    async def client_handler(self, websocket):
        """Handle individual client connections"""
        client_id = None
        try:
            # Client registration
            registration = await websocket.recv()
            reg_data = json.loads(registration)
            client_id = reg_data['client_id']
            task = reg_data['task']
            
            self.clients[client_id] = {
                'websocket': websocket,
                'task': task,
                'status': 'connected'
            }
            
            logger.info(f"✅ Client {client_id} ({task}) registered. Total: {len(self.clients)}")
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'welcome',
                'message': f'Welcome {client_id}!',
                'server_model': self.config.server_model
            }))
            
            # Main training loop
            await self.handle_client_training(client_id, websocket)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
    
    async def handle_client_training(self, client_id: str, websocket):
        """Handle training rounds with a specific client"""
        
        while len(self.clients) < self.config.max_clients:
            await websocket.send(json.dumps({
                'type': 'status',
                'message': f'Connected: {len(self.clients)}/{self.config.max_clients}'
            }))
            await asyncio.sleep(1)
        
        logger.info(f"✅ Starting training with {len(self.clients)} clients")
        
        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"🚀 Starting round {round_num}")
            
            # Send training start signal
            await websocket.send(json.dumps({
                'type': 'train_start',
                'round': round_num,
                'server_parameters': self.serialize_parameters(self.server_model.get_trainable_parameters())
            }))
            
            # Wait for client results
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=self.config.client_timeout)
                result_data = json.loads(response)
                
                if result_data['type'] == 'train_complete':
                    # Store client results
                    self.client_models[client_id] = {
                        'parameters': result_data['parameters'],
                        'metrics': result_data['metrics'],
                        'round': round_num
                    }
                    
                    logger.info(f"📊 {client_id}: Received results for round {round_num}")
                    
                    # Simple aggregation (in real implementation, wait for all clients)
                    # For demo purposes, we'll just acknowledge
                    await websocket.send(json.dumps({
                        'type': 'round_complete',
                        'round': round_num,
                        'message': f'Round {round_num} completed'
                    }))
                
            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} timed out in round {round_num}")
                break
        
        logger.info(f"🎉 Training completed for {client_id}")
    
    def serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """Serialize model parameters for transmission"""
        serialized = {}
        for name, param in parameters.items():
            serialized[name] = param.detach().cpu().numpy().tolist()
        return serialized
    
    def deserialize_parameters(self, serialized: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Deserialize parameters from client"""
        parameters = {}
        for name, param_list in serialized.items():
            parameters[name] = torch.tensor(param_list)
        return parameters
    
    async def start_server(self):
        """Start the federated server"""
        logger.info(f"🌐 Starting server on {self.config.server_host}:{self.config.server_port}")
        
        async with websockets.serve(
            self.client_handler,
            self.config.server_host,
            self.config.server_port
        ):
            logger.info("📡 Server started")
            logger.info("⏳ Waiting for clients...")
            await asyncio.Future()  # Run forever

class NoLoRAFederatedClient:
    """Federated client without LoRA - pure model training with knowledge distillation"""
    
    def __init__(self, client_id: str, task: str, config: NoLoRAConfig):
        self.client_id = client_id
        self.task = task.lower()
        self.config = config
        
        # Load task-specific data
        texts, labels = load_glue_data(self.task, config.max_samples_per_client)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)
        
        # Determine task configuration
        if self.task == 'stsb':
            self.model = NoLoRABERTModel(config.client_model, num_labels=1, task_type='regression')
        else:
            self.model = NoLoRABERTModel(config.client_model, num_labels=2, task_type='classification')
        
        # Create dataset and dataloader
        self.dataset = GLUEDataset(texts, labels, self.tokenizer, self.task)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Training setup
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model {config.client_model}: {total_params:,}/{total_params:,} params ({trainable_params/total_params*100:.1f}% trainable)")
        logger.info(f"Client {client_id} ({self.task}) initialized with {len(self.dataset)} samples")
    
    def train_local_epoch(self, server_parameters: Optional[Dict[str, torch.Tensor]] = None):
        """Train for one local epoch with optional knowledge distillation"""
        self.model.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        
        # Create teacher model if server parameters provided
        teacher_model = None
        if server_parameters:
            teacher_model = NoLoRABERTModel(
                self.config.server_model,
                num_labels=2 if self.task != 'stsb' else 1,
                task_type='regression' if self.task == 'stsb' else 'classification'
            )
            teacher_model.set_parameters(server_parameters)
            teacher_model.to(self.device)
            teacher_model.eval()
        
        for batch in self.dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Student forward pass
            student_outputs = self.model(input_ids, attention_mask, labels)
            
            if teacher_model is not None:
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids, attention_mask)
                
                # Knowledge distillation loss
                loss, kd_loss, task_loss = knowledge_distillation_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                    self.config.distillation_temperature,
                    self.config.distillation_alpha,
                    self.model.task_type
                )
                
                total_kd_loss += kd_loss.item()
                total_task_loss += task_loss.item()
            else:
                # Regular training loss
                loss = student_outputs.loss
                total_task_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_kd_loss = total_kd_loss / num_batches if num_batches > 0 else 0
        avg_task_loss = total_task_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'kd_loss': avg_kd_loss,
            'task_loss': avg_task_loss
        }
    
    def evaluate(self):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                total_loss += outputs.loss.item()
                
                if self.task != 'stsb':  # Classification
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
        
        avg_loss = total_loss / len(self.dataloader)
        
        if self.task == 'stsb':  # Regression
            return {'loss': avg_loss, 'mse': avg_loss}
        else:  # Classification
            accuracy = correct / total if total > 0 else 0
            return {'loss': avg_loss, 'accuracy': accuracy}
    
    def serialize_parameters(self) -> Dict[str, List]:
        """Serialize model parameters for transmission"""
        parameters = self.model.get_trainable_parameters()
        serialized = {}
        for name, param in parameters.items():
            serialized[name] = param.detach().cpu().numpy().tolist()
        return serialized
    
    async def run_client(self):
        """Main client execution loop"""
        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        
        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"🔗 {self.client_id} connected to server")
                
                # Register with server
                registration = {
                    'client_id': self.client_id,
                    'task': self.task,
                    'model': self.config.client_model
                }
                await websocket.send(json.dumps(registration))
                
                # Wait for welcome message
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                logger.info(f"🎉 {self.client_id}: {welcome_data['message']}")
                
                # Training loop
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data['type'] == 'status':
                            logger.info(f"📊 {self.client_id}: {data['message']}")
                            
                        elif data['type'] == 'train_start':
                            round_num = data['round']
                            server_params = data.get('server_parameters')
                            
                            logger.info(f"🎯 {self.client_id}: Starting round {round_num}")
                            
                            # Deserialize server parameters if provided
                            server_parameters = None
                            if server_params:
                                server_parameters = {}
                                for name, param_list in server_params.items():
                                    server_parameters[name] = torch.tensor(param_list)
                            
                            # Local training
                            for epoch in range(self.config.local_epochs):
                                train_metrics = self.train_local_epoch(server_parameters)
                                logger.info(f"📈 {self.client_id}: Epoch {epoch+1}, Loss: {train_metrics['loss']:.4f}")
                            
                            # Evaluation
                            eval_metrics = self.evaluate()
                            
                            # Send results back
                            result = {
                                'type': 'train_complete',
                                'round': round_num,
                                'parameters': self.serialize_parameters(),
                                'metrics': {
                                    'train': train_metrics,
                                    'eval': eval_metrics
                                }
                            }
                            await websocket.send(json.dumps(result))
                            logger.info(f"📊 {self.client_id}: Sent results for round {round_num}")
                            
                        elif data['type'] == 'round_complete':
                            logger.info(f"✅ {self.client_id}: {data['message']}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"🔌 {self.client_id}: Connection closed")
                        break
                        
        except Exception as e:
            logger.error(f"❌ {self.client_id}: Connection failed: {e}")
        finally:
            logger.info(f"🔌 {self.client_id}: Disconnected")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Streaming Federated Learning WITHOUT LoRA")
    parser.add_argument("--mode", choices=["server", "client"], required=True, help="Run as server or client")
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--task", choices=["sst2", "qqp", "stsb"], help="GLUE task (required for client mode)")
    parser.add_argument("--port", type=int, default=8768, help="Server port")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--samples", type=int, default=100, help="Max samples per client")
    
    args = parser.parse_args()
    
    # Configuration
    config = NoLoRAConfig(
        server_port=args.port,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        max_samples_per_client=args.samples
    )
    
    print("🚀 Streaming Federated Learning WITHOUT LoRA")
    print(f"Mode: {args.mode}")
    print(f"Server Model: {config.server_model}")
    print(f"Client Model: {config.client_model}")
    print(f"Port: {config.server_port}")
    print(f"Rounds: {config.num_rounds}")
    print("-" * 50)
    
    if args.mode == "server":
        print("🌐 Starting Federated Server...")
        server = NoLoRAFederatedServer(config)
        await server.start_server()
        
    elif args.mode == "client":
        if not args.client_id or not args.task:
            print("❌ Client mode requires --client_id and --task arguments")
            return
            
        print(f"👤 Starting client: {args.client_id} ({args.task.upper()})")
        client = NoLoRAFederatedClient(args.client_id, args.task, config)
        await client.run_client()

if __name__ == "__main__":
    asyncio.run(main())
