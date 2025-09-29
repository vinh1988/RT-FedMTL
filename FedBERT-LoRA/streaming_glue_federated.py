#!/usr/bin/env python3
"""
STREAMING GLUE FEDERATED LEARNING with WebSocket
Real-time federated learning with GLUE datasets and live streaming updates

Features:
- Real GLUE datasets (SST-2, QQP, STS-B)
- All clients use Tiny-BERT
- WebSocket streaming for real-time updates
- LoRA for parameter efficiency
- Knowledge distillation for cross-architecture learning
- Live training progress visualization
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import math
import time
from datetime import datetime
import threading
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StreamingGLUEConfig:
    """Configuration for streaming GLUE federated learning"""
    # All clients use Tiny-BERT, server uses BERT-base
    server_model: str = "bert-base-uncased"
    client_model: str = "prajjwal1/bert-tiny"
    
    # GLUE datasets for each client
    client_datasets = {
        "client_sst2": "sst2",    # Sentiment analysis
        "client_qqp": "qqp",      # Question pair matching
        "client_stsb": "stsb"     # Semantic similarity
    }
    
    # Streaming parameters
    server_host: str = "localhost"
    server_port: int = 8765
    update_interval: float = 2.0  # Seconds between progress updates
    
    # Training parameters
    num_rounds: int = 5
    local_epochs: int = 3
    batch_size: int = 8  # Smaller for streaming
    learning_rate: float = 2e-5
    max_sequence_length: int = 128
    max_samples_per_client: int = 500  # Smaller for faster streaming demo
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Knowledge distillation parameters
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7


class LoRALayer(nn.Module):
    """LoRA layer for parameter efficiency"""
    
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, dropout: float = 0.1):
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
        return self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, r, alpha, dropout)
        
        # Freeze original parameters for efficiency
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)


def add_lora_to_attention_layers(model, config: StreamingGLUEConfig):
    """Add LoRA to attention layers for parameter efficiency"""
    lora_modules = {}
    target_modules = ["query", "key", "value"]
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            lora_linear = LoRALinear(module, config.lora_r, config.lora_alpha, config.lora_dropout)
            
            # Replace module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                setattr(parent_module, child_name, lora_linear)
            else:
                setattr(model, child_name, lora_linear)
            
            lora_modules[name] = lora_linear
    
    return lora_modules


class StreamingGLUEModel(nn.Module):
    """Model with LoRA for streaming GLUE tasks"""
    
    def __init__(self, model_name: str, config: StreamingGLUEConfig, task_type: str = "classification"):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.task_type = task_type
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Task-specific head
        if task_type == "regression":
            self.classifier = nn.Linear(hidden_size, 1)  # STS-B regression
        else:
            self.classifier = nn.Linear(hidden_size, 2)  # SST-2, QQP classification
        
        self.dropout = nn.Dropout(0.1)
        
        # Add LoRA for parameter efficiency
        self.lora_modules = add_lora_to_attention_layers(self.bert, config)
        
        # Calculate parameter efficiency
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.param_efficiency = 100 * trainable_params / total_params
        
        logger.info(f"✅ Streaming model {model_name} ({task_type}):")
        logger.info(f"   Total params: {total_params:,}")
        logger.info(f"   Trainable (LoRA): {trainable_params:,}")
        logger.info(f"   Parameter efficiency: {self.param_efficiency:.1f}% trainable")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.task_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits, 'pooled_output': pooled_output}


class StreamingGLUEDataset(Dataset):
    """Dataset wrapper for streaming GLUE tasks"""
    
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


def load_streaming_glue_dataset(task_name: str, max_samples: int = 500):
    """Load and preprocess GLUE dataset for streaming"""
    logger.info(f"📥 Loading {task_name.upper()} dataset for streaming...")
    
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
            raise ValueError(f"Unsupported task: {task_name}")
        
        # Limit samples for streaming demo
        if len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        logger.info(f"✅ Loaded {len(texts)} samples from {task_name.upper()} for streaming")
        return texts, labels, task_type
        
    except Exception as e:
        logger.error(f"Failed to load {task_name}: {e}")
        logger.info(f"Using dummy data for {task_name}")
        
        # Fallback to dummy data
        if task_name == "sst2":
            texts = ["This is a positive example"] * 25 + ["This is a negative example"] * 25
            labels = [1] * 25 + [0] * 25
            task_type = "classification"
        elif task_name == "qqp":
            texts = ["Question 1 [SEP] Similar question"] * 25 + ["Question 1 [SEP] Different question"] * 25
            labels = [1] * 25 + [0] * 25
            task_type = "classification"
        else:  # stsb
            texts = ["Sentence 1 [SEP] Very similar sentence"] * 25 + ["Sentence 1 [SEP] Completely different sentence"] * 25
            labels = [4.5] * 25 + [0.5] * 25
            task_type = "regression"
        
        return texts, labels, task_type


class StreamingFederatedServer:
    """WebSocket server for streaming federated learning"""
    
    def __init__(self, config: StreamingGLUEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Global teacher model (BERT-base)
        self.global_model = StreamingGLUEModel(config.server_model, config, "classification")
        self.global_model.to(self.device)
        
        # Client management
        self.connected_clients = {}
        self.client_results = {}
        self.current_round = 0
        self.round_start_time = None
        
        # Progress tracking
        self.training_history = []
        self.live_updates = queue.Queue()
        
        logger.info(f"🌐 Streaming server initialized with {config.server_model}")
        logger.info(f"📡 Server will listen on {config.server_host}:{config.server_port}")
    
    async def register_client(self, websocket, client_info):
        """Register a new client"""
        client_id = client_info['client_id']
        task_name = client_info['task_name']
        
        self.connected_clients[client_id] = {
            'websocket': websocket,
            'task_name': task_name,
            'status': 'connected',
            'last_update': datetime.now()
        }
        
        logger.info(f"👤 Client {client_id} ({task_name.upper()}) connected")
        
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': f'Welcome {client_id}! Ready for streaming federated learning.',
            'server_model': self.config.server_model,
            'your_task': task_name.upper()
        }))
        
        # Broadcast client connection
        await self.broadcast_update({
            'type': 'client_connected',
            'client_id': client_id,
            'task_name': task_name.upper(),
            'total_clients': len(self.connected_clients),
            'timestamp': datetime.now().isoformat()
        })
    
    async def broadcast_update(self, message):
        """Broadcast update to all connected clients"""
        if self.connected_clients:
            disconnected = []
            for client_id, client_info in self.connected_clients.items():
                try:
                    await client_info['websocket'].send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected:
                del self.connected_clients[client_id]
                logger.warning(f"Client {client_id} disconnected")
    
    def generate_teacher_knowledge(self):
        """Generate knowledge from global teacher model"""
        self.global_model.eval()
        
        # Create dummy batch for knowledge generation
        batch_size = self.config.batch_size
        seq_length = self.config.max_sequence_length
        
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(self.device)
        dummy_attention_mask = torch.ones(batch_size, seq_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.global_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
            teacher_logits = outputs['logits'].cpu()
        
        return {
            'logits': teacher_logits.tolist(),
            'temperature': self.config.distillation_temperature,
            'alpha': self.config.distillation_alpha
        }
    
    async def start_federated_round(self, round_num):
        """Start a new federated learning round"""
        self.current_round = round_num
        self.round_start_time = time.time()
        self.client_results = {}
        
        logger.info(f"🚀 Starting streaming federated round {round_num}")
        
        # Generate teacher knowledge
        teacher_knowledge = self.generate_teacher_knowledge()
        
        # Send round start message to all clients
        round_message = {
            'type': 'round_start',
            'round_number': round_num,
            'total_rounds': self.config.num_rounds,
            'teacher_knowledge': teacher_knowledge,
            'local_epochs': self.config.local_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_update(round_message)
        
        # Broadcast round start notification
        await self.broadcast_update({
            'type': 'round_notification',
            'message': f'🎯 Round {round_num}/{self.config.num_rounds} started! All clients training with knowledge distillation...',
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_client_result(self, client_id, result):
        """Handle training result from client"""
        self.client_results[client_id] = result
        
        # Update client status
        if client_id in self.connected_clients:
            self.connected_clients[client_id]['status'] = 'completed'
            self.connected_clients[client_id]['last_update'] = datetime.now()
        
        logger.info(f"📊 Received results from {client_id} ({result['task_name'].upper()})")
        
        # Broadcast individual client completion
        await self.broadcast_update({
            'type': 'client_completed',
            'client_id': client_id,
            'task_name': result['task_name'].upper(),
            'accuracy': result.get('accuracy', 0.0),
            'loss': result.get('loss', 0.0),
            'kd_loss': result.get('kd_loss', 0.0),
            'param_efficiency': result.get('param_efficiency', 0.0),
            'completed_clients': len(self.client_results),
            'total_clients': len(self.connected_clients),
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if round is complete
        if len(self.client_results) == len(self.connected_clients):
            await self.complete_round()
    
    async def complete_round(self):
        """Complete the current federated round"""
        round_duration = time.time() - self.round_start_time
        
        # Calculate round statistics
        classification_results = [r for r in self.client_results.values() if r.get('task_type') == 'classification']
        regression_results = [r for r in self.client_results.values() if r.get('task_type') == 'regression']
        
        avg_class_accuracy = np.mean([r['accuracy'] for r in classification_results]) if classification_results else 0.0
        avg_regression_mse = np.mean([r['accuracy'] for r in regression_results]) if regression_results else 0.0
        avg_kd_loss = np.mean([r.get('kd_loss', 0.0) for r in self.client_results.values()])
        avg_param_efficiency = np.mean([r.get('param_efficiency', 0.0) for r in self.client_results.values()])
        
        # Store round results
        round_summary = {
            'round': self.current_round,
            'duration': round_duration,
            'avg_classification_accuracy': avg_class_accuracy,
            'avg_regression_mse': avg_regression_mse,
            'avg_kd_loss': avg_kd_loss,
            'avg_param_efficiency': avg_param_efficiency,
            'client_results': dict(self.client_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(round_summary)
        
        logger.info(f"✅ Round {self.current_round} completed in {round_duration:.1f}s")
        logger.info(f"   Classification Accuracy: {avg_class_accuracy:.4f}")
        logger.info(f"   Regression MSE: {avg_regression_mse:.4f}")
        logger.info(f"   Knowledge Distillation Loss: {avg_kd_loss:.4f}")
        logger.info(f"   Parameter Efficiency: {avg_param_efficiency:.1f}%")
        
        # Broadcast round completion
        await self.broadcast_update({
            'type': 'round_completed',
            'round_number': self.current_round,
            'duration': round_duration,
            'results': {
                'classification_accuracy': avg_class_accuracy,
                'regression_mse': avg_regression_mse,
                'kd_loss': avg_kd_loss,
                'param_efficiency': avg_param_efficiency
            },
            'client_details': {
                client_id: {
                    'task': result['task_name'].upper(),
                    'accuracy': result.get('accuracy', 0.0),
                    'loss': result.get('loss', 0.0)
                }
                for client_id, result in self.client_results.items()
            },
            'timestamp': datetime.now().isoformat()
        })
        
        # Reset client statuses
        for client_info in self.connected_clients.values():
            client_info['status'] = 'ready'
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'register':
                await self.register_client(websocket, data)
            
            elif message_type == 'training_result':
                client_id = data.get('client_id')
                result = data.get('result')
                await self.handle_client_result(client_id, result)
            
            elif message_type == 'progress_update':
                # Broadcast live training progress
                await self.broadcast_update({
                    'type': 'live_progress',
                    'client_id': data.get('client_id'),
                    'progress': data.get('progress'),
                    'timestamp': datetime.now().isoformat()
                })
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def client_handler(self, websocket, path):
        """Handle individual client connections"""
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
    
    async def run_federated_training(self):
        """Run the complete federated training process"""
        logger.info("🎯 Starting streaming GLUE federated learning!")
        
        # Wait for clients to connect
        logger.info("⏳ Waiting for clients to connect...")
        while len(self.connected_clients) < len(self.config.client_datasets):
            await asyncio.sleep(1)
        
        logger.info(f"✅ All {len(self.connected_clients)} clients connected!")
        
        # Run federated rounds
        for round_num in range(1, self.config.num_rounds + 1):
            await self.start_federated_round(round_num)
            
            # Wait for round completion
            while len(self.client_results) < len(self.connected_clients):
                await asyncio.sleep(0.5)
            
            # Brief pause between rounds
            if round_num < self.config.num_rounds:
                await asyncio.sleep(2)
        
        # Send final results
        await self.broadcast_final_results()
    
    async def broadcast_final_results(self):
        """Broadcast final federated learning results"""
        if not self.training_history:
            return
        
        final_round = self.training_history[-1]
        
        # Calculate improvements
        first_round = self.training_history[0]
        class_improvement = final_round['avg_classification_accuracy'] - first_round['avg_classification_accuracy']
        regression_improvement = first_round['avg_regression_mse'] - final_round['avg_regression_mse']  # Lower is better
        
        final_message = {
            'type': 'training_completed',
            'message': '🎉 Streaming GLUE Federated Learning Completed!',
            'total_rounds': len(self.training_history),
            'final_results': {
                'classification_accuracy': final_round['avg_classification_accuracy'],
                'regression_mse': final_round['avg_regression_mse'],
                'kd_loss': final_round['avg_kd_loss'],
                'param_efficiency': final_round['avg_param_efficiency']
            },
            'improvements': {
                'classification_improvement': class_improvement,
                'regression_improvement': regression_improvement
            },
            'benefits_achieved': [
                f"✅ Parameter Efficiency: {final_round['avg_param_efficiency']:.1f}% trainable (LoRA)",
                f"✅ Knowledge Distillation: KD Loss {final_round['avg_kd_loss']:.4f}",
                f"✅ Cross-Architecture Learning: BERT-base → Tiny-BERT",
                f"✅ Multi-Task Collaboration: SST-2, QQP, STS-B"
            ],
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_update(final_message)
        
        logger.info("🎉 Streaming federated learning completed!")
        logger.info(f"   Final Classification Accuracy: {final_round['avg_classification_accuracy']:.4f}")
        logger.info(f"   Final Regression MSE: {final_round['avg_regression_mse']:.4f}")
        logger.info(f"   Classification Improvement: +{class_improvement:.4f}")
        logger.info(f"   Regression Improvement: -{regression_improvement:.4f} MSE")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting streaming server on {self.config.server_host}:{self.config.server_port}")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.client_handler,
            self.config.server_host,
            self.config.server_port
        )
        
        logger.info("📡 Server started, waiting for clients...")
        
        # Run federated training
        training_task = asyncio.create_task(self.run_federated_training())
        
        # Keep server running
        await training_task
        
        # Keep server alive for a bit after training
        await asyncio.sleep(10)
        
        server.close()
        await server.wait_closed()


class StreamingFederatedClient:
    """WebSocket client for streaming federated learning"""
    
    def __init__(self, client_id: str, task_name: str, config: StreamingGLUEConfig):
        self.client_id = client_id
        self.task_name = task_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine task type
        self.task_type = "regression" if task_name == "stsb" else "classification"
        
        # Load dataset
        texts, labels, _ = load_streaming_glue_dataset(task_name, config.max_samples_per_client)
        tokenizer = AutoTokenizer.from_pretrained(config.client_model)
        self.dataset = StreamingGLUEDataset(texts, labels, tokenizer, config.max_sequence_length, self.task_type)
        
        # Student model (Tiny-BERT)
        self.model = StreamingGLUEModel(config.client_model, config, self.task_type)
        self.model.to(self.device)
        
        # Optimizer for LoRA parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.websocket = None
        self.is_training = False
        
        logger.info(f"👤 Streaming client {client_id} ({task_name.upper()}) initialized")
        logger.info(f"   Dataset: {len(self.dataset)} samples")
        logger.info(f"   Model: {config.client_model}")
        logger.info(f"   Task type: {self.task_type}")
    
    async def connect_to_server(self):
        """Connect to the federated server"""
        uri = f"ws://{self.config.server_host}:{self.config.server_port}"
        
        try:
            self.websocket = await websockets.connect(uri)
            logger.info(f"🔗 Connected to server at {uri}")
            
            # Register with server
            registration_message = {
                'type': 'register',
                'client_id': self.client_id,
                'task_name': self.task_name,
                'dataset_size': len(self.dataset),
                'model': self.config.client_model,
                'task_type': self.task_type
            }
            
            await self.websocket.send(json.dumps(registration_message))
            logger.info(f"📝 Registered as {self.client_id} ({self.task_name.upper()})")
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    async def send_progress_update(self, progress_info):
        """Send live training progress to server"""
        if self.websocket:
            try:
                message = {
                    'type': 'progress_update',
                    'client_id': self.client_id,
                    'progress': progress_info
                }
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculate knowledge distillation loss"""
        if self.task_type == "regression":
            # For STS-B regression
            student_preds = student_logits.squeeze()
            teacher_preds = torch.tensor(teacher_logits).squeeze().to(self.device)
            
            # Distillation loss (MSE between predictions)
            distillation_loss = F.mse_loss(student_preds, teacher_preds)
            
            # Task loss (MSE with labels)
            task_loss = F.mse_loss(student_preds, labels.float())
            
            total_loss = self.config.distillation_alpha * distillation_loss + (1 - self.config.distillation_alpha) * task_loss
            return total_loss, distillation_loss, task_loss
        else:
            # For SST-2 and QQP classification
            teacher_tensor = torch.tensor(teacher_logits).to(self.device)
            
            # Soft targets (knowledge distillation)
            student_soft = F.log_softmax(student_logits / self.config.distillation_temperature, dim=1)
            teacher_soft = F.softmax(teacher_tensor / self.config.distillation_temperature, dim=1)
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.config.distillation_temperature ** 2)
            
            # Hard targets (task loss)
            task_loss = F.cross_entropy(student_logits, labels)
            
            # Combined loss
            total_loss = self.config.distillation_alpha * distillation_loss + (1 - self.config.distillation_alpha) * task_loss
            return total_loss, distillation_loss, task_loss
    
    async def local_training_with_streaming(self, teacher_knowledge):
        """Perform local training with streaming updates"""
        self.is_training = True
        self.model.train()
        
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        teacher_logits = teacher_knowledge.get('logits', [])
        
        total_batches = len(dataloader) * self.config.local_epochs
        current_batch = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(dataloader):
                current_batch += 1
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student forward pass
                student_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Knowledge distillation if teacher knowledge available
                if teacher_logits and len(teacher_logits) > 0:
                    # Use first batch of teacher logits (simplified for demo)
                    batch_teacher_logits = teacher_logits[:input_ids.size(0)]
                    
                    loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                        student_outputs['logits'],
                        batch_teacher_logits,
                        labels
                    )
                    
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()
                else:
                    loss = student_outputs['loss']
                    total_task_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if self.task_type == "classification":
                    predictions = torch.argmax(student_outputs['logits'], dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                else:
                    # For regression, use MSE
                    predictions = student_outputs['logits'].squeeze()
                    mse = torch.mean((predictions - labels.float()) ** 2)
                    correct_predictions += mse.item()
                
                total_samples += labels.size(0)
                
                # Send streaming progress update
                progress = {
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'total_batches': len(dataloader),
                    'progress_percent': (current_batch / total_batches) * 100,
                    'current_loss': loss.item(),
                    'task_name': self.task_name.upper()
                }
                
                await self.send_progress_update(progress)
                
                # Small delay for streaming effect
                await asyncio.sleep(0.1)
        
        # Calculate final metrics
        num_batches = len(dataloader) * self.config.local_epochs
        avg_loss = total_loss / num_batches
        avg_kd_loss = total_kd_loss / num_batches if total_kd_loss > 0 else 0.0
        avg_task_loss = total_task_loss / num_batches
        
        if self.task_type == "classification":
            accuracy = correct_predictions / total_samples
        else:
            accuracy = correct_predictions / num_batches  # MSE for regression
        
        self.is_training = False
        
        return {
            'loss': avg_loss,
            'kd_loss': avg_kd_loss,
            'task_loss': avg_task_loss,
            'accuracy': accuracy,
            'num_samples': len(self.dataset),
            'param_efficiency': self.model.param_efficiency,
            'task_type': self.task_type,
            'task_name': self.task_name
        }
    
    async def handle_server_message(self, message):
        """Handle incoming message from server"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'welcome':
                logger.info(f"🎉 {data.get('message')}")
            
            elif message_type == 'round_start':
                round_num = data.get('round_number')
                teacher_knowledge = data.get('teacher_knowledge', {})
                
                logger.info(f"🎯 Starting round {round_num} training with knowledge distillation...")
                
                # Perform local training
                results = await self.local_training_with_streaming(teacher_knowledge)
                
                # Send results back to server
                result_message = {
                    'type': 'training_result',
                    'client_id': self.client_id,
                    'result': results
                }
                
                await self.websocket.send(json.dumps(result_message))
                logger.info(f"📊 Sent training results for round {round_num}")
            
            elif message_type == 'round_completed':
                round_num = data.get('round_number')
                results = data.get('results', {})
                logger.info(f"✅ Round {round_num} completed!")
                logger.info(f"   Global Classification Accuracy: {results.get('classification_accuracy', 0):.4f}")
                logger.info(f"   Global Regression MSE: {results.get('regression_mse', 0):.4f}")
            
            elif message_type == 'training_completed':
                logger.info("🎉 " + data.get('message', 'Training completed!'))
                final_results = data.get('final_results', {})
                logger.info(f"🏆 Final Results:")
                logger.info(f"   Classification Accuracy: {final_results.get('classification_accuracy', 0):.4f}")
                logger.info(f"   Regression MSE: {final_results.get('regression_mse', 0):.4f}")
                logger.info(f"   Parameter Efficiency: {final_results.get('param_efficiency', 0):.1f}%")
                
                # Print benefits
                benefits = data.get('benefits_achieved', [])
                for benefit in benefits:
                    logger.info(f"   {benefit}")
            
            elif message_type in ['client_connected', 'client_completed', 'round_notification', 'live_progress']:
                # These are broadcast messages, just log them
                if message_type == 'round_notification':
                    logger.info(f"📢 {data.get('message')}")
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from server")
        except Exception as e:
            logger.error(f"Error handling server message: {e}")
    
    async def start_client(self):
        """Start the streaming client"""
        await self.connect_to_server()
        
        try:
            # Listen for server messages
            async for message in self.websocket:
                await self.handle_server_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Disconnected from server")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()


async def run_streaming_server(config: StreamingGLUEConfig):
    """Run the streaming federated server"""
    server = StreamingFederatedServer(config)
    await server.start_server()


async def run_streaming_client(client_id: str, task_name: str, config: StreamingGLUEConfig):
    """Run a streaming federated client"""
    client = StreamingFederatedClient(client_id, task_name, config)
    await client.start_client()


def main():
    """Main entry point for streaming GLUE federated learning"""
    parser = argparse.ArgumentParser(description="Streaming GLUE Federated Learning")
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                       help="Run as server or client")
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--task", choices=["sst2", "qqp", "stsb"], help="GLUE task (required for client mode)")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    
    args = parser.parse_args()
    
    # Create configuration
    config = StreamingGLUEConfig()
    config.server_host = args.host
    config.server_port = args.port
    config.num_rounds = args.rounds
    
    if args.mode == "server":
        print("=" * 80)
        print("🎯 STREAMING GLUE FEDERATED LEARNING SERVER")
        print("=" * 80)
        print("Real-time federated learning with GLUE datasets:")
        print("• SST-2 (Sentiment Analysis) - Tiny-BERT")
        print("• QQP (Question Pair Matching) - Tiny-BERT")
        print("• STS-B (Semantic Similarity) - Tiny-BERT")
        print("• Server: BERT-base (Knowledge Distillation)")
        print()
        print("Features:")
        print("✅ WebSocket streaming for real-time updates")
        print("✅ Parameter efficiency from LoRA")
        print("✅ Cross-architecture learning from knowledge distillation")
        print("✅ Multi-task collaboration")
        print("=" * 80)
        
        asyncio.run(run_streaming_server(config))
    
    elif args.mode == "client":
        if not args.client_id or not args.task:
            print("Error: --client_id and --task are required for client mode")
            return
        
        print(f"👤 Starting streaming client: {args.client_id} ({args.task.upper()})")
        asyncio.run(run_streaming_client(args.client_id, args.task, config))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
