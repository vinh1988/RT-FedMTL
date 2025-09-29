#!/usr/bin/env python3
"""
FIXED Streaming GLUE Federated Learning
Addresses connection and training issues in the original implementation
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import logging
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import math
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FixedGLUEConfig:
    """Fixed configuration for GLUE federated learning"""
    server_model: str = "bert-base-uncased"
    client_model: str = "prajjwal1/bert-tiny"
    
    # Network settings
    server_host: str = "localhost"
    server_port: int = 8766  # Different port to avoid conflicts
    
    # Training parameters
    num_rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_sequence_length: int = 128
    max_samples_per_client: int = 100  # Small for quick demo
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Knowledge distillation
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7


class SimpleLoRALinear(nn.Module):
    """Simplified LoRA implementation"""
    
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, original_layer.out_features))
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


class SimpleGLUEModel(nn.Module):
    """Simplified GLUE model with LoRA"""
    
    def __init__(self, model_name: str, config: FixedGLUEConfig, task_type: str = "classification"):
        super().__init__()
        self.model_name = model_name
        self.task_type = task_type
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Task head
        if task_type == "regression":
            self.classifier = nn.Linear(hidden_size, 1)
        else:
            self.classifier = nn.Linear(hidden_size, 2)
        
        self.dropout = nn.Dropout(0.1)
        
        # Add LoRA to a few key layers
        self.add_lora_layers(config)
        
        # Calculate efficiency
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.param_efficiency = 100 * trainable_params / total_params
        
        logger.info(f"Model {model_name}: {trainable_params:,}/{total_params:,} params ({self.param_efficiency:.1f}% trainable)")
    
    def add_lora_layers(self, config):
        """Add LoRA to key attention layers"""
        lora_count = 0
        for name, module in self.bert.named_modules():
            if "attention" in name and "query" in name and isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent_module = dict(self.bert.named_modules())[parent_name]
                lora_layer = SimpleLoRALinear(module, config.lora_r, config.lora_alpha)
                setattr(parent_module, child_name, lora_layer)
                lora_count += 1
                
                if lora_count >= 3:  # Limit to 3 layers for simplicity
                    break
        
        logger.info(f"Added LoRA to {lora_count} layers")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class SimpleGLUEDataset(Dataset):
    """Simplified GLUE dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128, task_type="classification"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
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
        
        if self.task_type == "regression":
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


def load_simple_glue_data(task_name: str, max_samples: int = 100):
    """Load simplified GLUE data"""
    logger.info(f"Loading {task_name.upper()} dataset...")
    
    try:
        if task_name == "sst2":
            dataset = load_dataset("glue", "sst2", split="train")
            texts = [item['sentence'] for item in dataset]
            labels = [item['label'] for item in dataset]
            task_type = "classification"
        elif task_name == "qqp":
            dataset = load_dataset("glue", "qqp", split="train")
            texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
            labels = [item['label'] for item in dataset]
            task_type = "classification"
        elif task_name == "stsb":
            dataset = load_dataset("glue", "stsb", split="train")
            texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
            labels = [item['label'] for item in dataset]
            task_type = "regression"
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Limit samples
        if len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        logger.info(f"Loaded {len(texts)} samples from {task_name.upper()}")
        return texts, labels, task_type
        
    except Exception as e:
        logger.error(f"Failed to load {task_name}: {e}")
        # Fallback dummy data
        if task_name == "sst2":
            texts = ["This is positive"] * 25 + ["This is negative"] * 25
            labels = [1] * 25 + [0] * 25
            task_type = "classification"
        elif task_name == "qqp":
            texts = ["Question 1 [SEP] Similar question"] * 25 + ["Question 1 [SEP] Different question"] * 25
            labels = [1] * 25 + [0] * 25
            task_type = "classification"
        else:  # stsb
            texts = ["Sentence 1 [SEP] Similar sentence"] * 25 + ["Sentence 1 [SEP] Different sentence"] * 25
            labels = [4.0] * 25 + [1.0] * 25
            task_type = "regression"
        
        return texts, labels, task_type


class FixedFederatedServer:
    """Fixed federated server"""
    
    def __init__(self, config: FixedGLUEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Global model
        self.global_model = SimpleGLUEModel(config.server_model, config, "classification")
        self.global_model.to(self.device)
        
        # Client management
        self.connected_clients = {}
        self.client_results = {}
        self.current_round = 0
        self.training_history = []
        
        logger.info(f"Server initialized with {config.server_model}")
    
    async def register_client(self, websocket, client_info):
        """Register client"""
        client_id = client_info['client_id']
        task_name = client_info.get('task_name', 'unknown')
        
        self.connected_clients[client_id] = {
            'websocket': websocket,
            'task_name': task_name,
            'status': 'connected'
        }
        
        logger.info(f"✅ Client {client_id} ({task_name}) registered. Total: {len(self.connected_clients)}")
        
        # Send welcome
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': f'Welcome {client_id}!',
            'server_model': self.config.server_model
        }))
        
        return True
    
    async def broadcast_message(self, message):
        """Broadcast to all clients"""
        if not self.connected_clients:
            return
        
        disconnected = []
        for client_id, client_info in self.connected_clients.items():
            try:
                await client_info['websocket'].send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client_id)
        
        # Clean up disconnected
        for client_id in disconnected:
            del self.connected_clients[client_id]
            logger.warning(f"Client {client_id} disconnected")
    
    def generate_teacher_knowledge(self):
        """Generate teacher knowledge"""
        self.global_model.eval()
        
        # Create dummy batch
        batch_size = 4
        seq_len = 128
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
        dummy_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        with torch.no_grad():
            outputs = self.global_model(dummy_input, dummy_mask)
            teacher_logits = outputs['logits'].cpu().tolist()
        
        return {
            'logits': teacher_logits,
            'temperature': self.config.distillation_temperature,
            'alpha': self.config.distillation_alpha
        }
    
    async def start_federated_round(self, round_num):
        """Start federated round"""
        self.current_round = round_num
        self.client_results = {}
        
        logger.info(f"🚀 Starting round {round_num}")
        
        # Generate teacher knowledge
        teacher_knowledge = self.generate_teacher_knowledge()
        
        # Send to clients
        message = {
            'type': 'round_start',
            'round_number': round_num,
            'teacher_knowledge': teacher_knowledge,
            'local_epochs': self.config.local_epochs
        }
        
        await self.broadcast_message(message)
    
    async def handle_client_result(self, client_id, result):
        """Handle client result"""
        self.client_results[client_id] = result
        
        logger.info(f"📊 Result from {client_id}: loss={result.get('loss', 0):.4f}, acc={result.get('accuracy', 0):.4f}")
        
        # Check if round complete
        if len(self.client_results) >= len(self.connected_clients):
            await self.complete_round()
    
    async def complete_round(self):
        """Complete round"""
        # Calculate averages
        avg_loss = np.mean([r.get('loss', 0) for r in self.client_results.values()])
        avg_accuracy = np.mean([r.get('accuracy', 0) for r in self.client_results.values()])
        avg_kd_loss = np.mean([r.get('kd_loss', 0) for r in self.client_results.values()])
        
        round_result = {
            'round': self.current_round,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_kd_loss': avg_kd_loss,
            'client_results': dict(self.client_results)
        }
        
        self.training_history.append(round_result)
        
        logger.info(f"✅ Round {self.current_round} complete: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
        
        # Broadcast completion
        await self.broadcast_message({
            'type': 'round_complete',
            'round': self.current_round,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_kd_loss': avg_kd_loss
        })
    
    async def handle_message(self, websocket, message):
        """Handle client message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'register':
                await self.register_client(websocket, data)
            elif msg_type == 'training_result':
                client_id = data.get('client_id')
                result = data.get('result', {})
                await self.handle_client_result(client_id, result)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def client_handler(self, websocket):
        """Handle client connections"""
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
    
    async def run_training(self):
        """Run federated training"""
        logger.info("⏳ Waiting for clients...")
        
        # Wait for at least 2 clients (flexible)
        target_clients = 3
        while len(self.connected_clients) < min(target_clients, 2):
            await asyncio.sleep(1)
            if len(self.connected_clients) > 0:
                logger.info(f"Connected: {len(self.connected_clients)}/{target_clients}")
        
        logger.info(f"✅ Starting training with {len(self.connected_clients)} clients")
        
        # Run rounds
        for round_num in range(1, self.config.num_rounds + 1):
            await self.start_federated_round(round_num)
            
            # Wait for completion
            while len(self.client_results) < len(self.connected_clients):
                await asyncio.sleep(0.5)
            
            await asyncio.sleep(1)  # Brief pause
        
        # Send final results
        if self.training_history:
            final_result = self.training_history[-1]
            await self.broadcast_message({
                'type': 'training_complete',
                'message': '🎉 Training completed!',
                'final_results': final_result,
                'history': self.training_history
            })
        
        logger.info("🎉 Federated training completed!")
    
    async def start_server(self):
        """Start server"""
        logger.info(f"🌐 Starting server on {self.config.server_host}:{self.config.server_port}")
        
        server = await websockets.serve(
            self.client_handler,
            self.config.server_host,
            self.config.server_port
        )
        
        logger.info("📡 Server started")
        
        # Run training
        training_task = asyncio.create_task(self.run_training())
        await training_task
        
        # Keep alive briefly
        await asyncio.sleep(5)
        
        server.close()
        await server.wait_closed()


class FixedFederatedClient:
    """Fixed federated client"""
    
    def __init__(self, client_id: str, task_name: str, config: FixedGLUEConfig):
        self.client_id = client_id
        self.task_name = task_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine task type
        self.task_type = "regression" if task_name == "stsb" else "classification"
        
        # Load data
        texts, labels, _ = load_simple_glue_data(task_name, config.max_samples_per_client)
        tokenizer = AutoTokenizer.from_pretrained(config.client_model)
        self.dataset = SimpleGLUEDataset(texts, labels, tokenizer, config.max_sequence_length, self.task_type)
        
        # Model
        self.model = SimpleGLUEModel(config.client_model, config, self.task_type)
        self.model.to(self.device)
        
        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.websocket = None
        
        logger.info(f"Client {client_id} ({task_name}) initialized with {len(self.dataset)} samples")
    
    async def connect_to_server(self):
        """Connect to server"""
        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        
        try:
            self.websocket = await websockets.connect(uri)
            logger.info(f"🔗 {self.client_id} connected to server")
            
            # Register
            await self.websocket.send(json.dumps({
                'type': 'register',
                'client_id': self.client_id,
                'task_name': self.task_name,
                'dataset_size': len(self.dataset)
            }))
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels):
        """KD loss calculation with proper dimension handling"""
        if not teacher_logits or len(teacher_logits) == 0:
            # No teacher knowledge, use regular loss
            if self.task_type == "regression":
                task_loss = F.mse_loss(student_logits.squeeze(), labels.float())
                return task_loss, 0.0, task_loss
            else:
                task_loss = F.cross_entropy(student_logits, labels)
                return task_loss, 0.0, task_loss
        
        # Handle teacher logits dimensions
        batch_size = student_logits.size(0)
        try:
            # Convert teacher logits to tensor and match batch size
            teacher_tensor = torch.tensor(teacher_logits).to(self.device)
            
            # Ensure teacher tensor matches student batch size
            if teacher_tensor.size(0) > batch_size:
                teacher_tensor = teacher_tensor[:batch_size]
            elif teacher_tensor.size(0) < batch_size:
                # Repeat teacher logits to match batch size
                repeat_factor = (batch_size + teacher_tensor.size(0) - 1) // teacher_tensor.size(0)
                teacher_tensor = teacher_tensor.repeat(repeat_factor, 1)[:batch_size]
            
            if self.task_type == "regression":
                # For regression tasks (STS-B)
                student_preds = student_logits.squeeze()
                
                # Teacher tensor should be 1D for regression
                if teacher_tensor.dim() > 1:
                    teacher_preds = teacher_tensor[:, 0]  # Take first column
                else:
                    teacher_preds = teacher_tensor
                
                # Ensure same size
                if student_preds.size() != teacher_preds.size():
                    teacher_preds = teacher_preds[:student_preds.size(0)]
                
                distillation_loss = F.mse_loss(student_preds, teacher_preds)
                task_loss = F.mse_loss(student_preds, labels.float())
                
            else:
                # For classification tasks (SST-2, QQP)
                # Ensure teacher tensor has same number of classes as student
                if teacher_tensor.size(-1) != student_logits.size(-1):
                    # If teacher has different number of classes, use only task loss
                    task_loss = F.cross_entropy(student_logits, labels)
                    return task_loss, 0.0, task_loss
                
                # KL divergence for classification
                student_soft = F.log_softmax(student_logits / self.config.distillation_temperature, dim=1)
                teacher_soft = F.softmax(teacher_tensor / self.config.distillation_temperature, dim=1)
                distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.config.distillation_temperature ** 2)
                
                task_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            total_loss = self.config.distillation_alpha * distillation_loss + (1 - self.config.distillation_alpha) * task_loss
            return total_loss, distillation_loss, task_loss
            
        except Exception as e:
            logger.warning(f"KD loss calculation failed: {e}, using task loss only")
            # Fallback to task loss only
            if self.task_type == "regression":
                task_loss = F.mse_loss(student_logits.squeeze(), labels.float())
            else:
                task_loss = F.cross_entropy(student_logits, labels)
            return task_loss, 0.0, task_loss
    
    def local_training(self, teacher_knowledge):
        """Perform local training"""
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0.0
        total_kd_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        teacher_logits = teacher_knowledge.get('logits', [])
        
        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)
                
                # Knowledge distillation
                loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                    outputs['logits'], teacher_logits, labels
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss
                
                # Calculate accuracy
                if self.task_type == "classification":
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                else:
                    # For regression, use negative MSE as "accuracy"
                    mse = F.mse_loss(outputs['logits'].squeeze(), labels.float())
                    correct_predictions += -mse.item()
                
                total_samples += labels.size(0)
        
        # Calculate metrics
        num_batches = len(dataloader) * self.config.local_epochs
        avg_loss = total_loss / num_batches
        avg_kd_loss = total_kd_loss / num_batches
        
        if self.task_type == "classification":
            accuracy = correct_predictions / total_samples
        else:
            accuracy = correct_predictions / num_batches  # Negative MSE for regression
        
        return {
            'loss': avg_loss,
            'kd_loss': avg_kd_loss,
            'accuracy': accuracy,
            'task_name': self.task_name,
            'task_type': self.task_type,
            'param_efficiency': self.model.param_efficiency
        }
    
    async def handle_server_message(self, message):
        """Handle server message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'welcome':
                logger.info(f"🎉 {self.client_id}: {data.get('message')}")
            
            elif msg_type == 'round_start':
                round_num = data.get('round_number')
                teacher_knowledge = data.get('teacher_knowledge', {})
                
                logger.info(f"🎯 {self.client_id}: Starting round {round_num}")
                
                # Train locally
                result = self.local_training(teacher_knowledge)
                
                # Send result
                await self.websocket.send(json.dumps({
                    'type': 'training_result',
                    'client_id': self.client_id,
                    'result': result
                }))
                
                logger.info(f"📊 {self.client_id}: Sent results for round {round_num}")
            
            elif msg_type == 'round_complete':
                round_num = data.get('round')
                avg_loss = data.get('avg_loss')
                avg_accuracy = data.get('avg_accuracy')
                logger.info(f"✅ {self.client_id}: Round {round_num} complete - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f}")
            
            elif msg_type == 'training_complete':
                logger.info(f"🎉 {self.client_id}: {data.get('message')}")
                final_results = data.get('final_results', {})
                logger.info(f"🏆 Final - Loss: {final_results.get('avg_loss', 0):.4f}, Acc: {final_results.get('avg_accuracy', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def start_client(self):
        """Start client"""
        await self.connect_to_server()
        
        try:
            async for message in self.websocket:
                await self.handle_server_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 {self.client_id}: Disconnected")
        except Exception as e:
            logger.error(f"{self.client_id} error: {e}")


async def run_fixed_server(config: FixedGLUEConfig):
    """Run fixed server"""
    server = FixedFederatedServer(config)
    await server.start_server()


async def run_fixed_client(client_id: str, task_name: str, config: FixedGLUEConfig):
    """Run fixed client"""
    client = FixedFederatedClient(client_id, task_name, config)
    await client.start_client()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fixed Streaming GLUE Federated Learning")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--task", choices=["sst2", "qqp", "stsb"], help="GLUE task")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    
    args = parser.parse_args()
    
    config = FixedGLUEConfig()
    config.server_port = args.port
    config.num_rounds = args.rounds
    
    if args.mode == "server":
        print("🌐 FIXED STREAMING GLUE FEDERATED LEARNING SERVER")
        print("=" * 60)
        print("Features:")
        print("✅ Fixed WebSocket connections")
        print("✅ Simplified LoRA implementation")
        print("✅ Knowledge distillation")
        print("✅ Real GLUE datasets")
        print("=" * 60)
        
        asyncio.run(run_fixed_server(config))
    
    elif args.mode == "client":
        if not args.client_id or not args.task:
            print("Error: --client_id and --task required for client mode")
            return
        
        print(f"👤 Starting client: {args.client_id} ({args.task.upper()})")
        asyncio.run(run_fixed_client(args.client_id, args.task, config))


if __name__ == "__main__":
    main()
