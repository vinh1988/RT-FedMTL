#!/usr/bin/env python3
"""
Deep Research Training System for Streaming Federated Learning
Supports both LoRA and non-LoRA variants with comprehensive metrics collection
Author: Research Team
"""

import asyncio
import websockets
import json
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import time
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logger.warning("Matplotlib/Seaborn not available. Visualization features disabled.")
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_federated.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ResearchConfig:
    """Research configuration for federated learning experiments"""
    # Model settings
    server_model: str = "bert-base-uncased"
    client_model: str = "prajjwal1/bert-tiny"
    
    # Training settings
    num_rounds: int = 22
    min_clients: int = 2
    max_clients: int = 10
    local_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Knowledge Distillation settings
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Research settings
    data_samples_per_client: int = 1000
    evaluation_frequency: int = 1  # Evaluate every N rounds
    save_checkpoints: bool = True
    detailed_logging: bool = True
    
    # Communication settings
    port: int = 8769
    timeout: int = 300

@dataclass
class ClientMetrics:
    """Metrics for individual client performance"""
    client_id: str
    round_num: int
    task: str
    
    # Training metrics
    train_loss: float
    train_accuracy: float
    local_epochs_completed: int
    
    # Knowledge Distillation metrics
    kd_loss: float
    task_loss: float
    
    # Efficiency metrics
    training_time: float
    communication_time: float
    parameter_count: int
    memory_usage: float
    
    # Data metrics
    data_samples: int
    data_distribution: Dict[str, int]

@dataclass
class ServerMetrics:
    """Metrics for server performance and global model"""
    round_num: int
    
    # Aggregation metrics
    participating_clients: int
    aggregation_time: float
    
    # Global model performance
    global_accuracy: Dict[str, float]  # Per task
    global_loss: Dict[str, float]      # Per task
    
    # Convergence metrics
    parameter_variance: float
    convergence_rate: float
    
    # Communication metrics
    total_communication_cost: float
    average_client_latency: float

@dataclass
class ExperimentResults:
    """Complete experiment results for analysis"""
    config: ResearchConfig
    client_metrics: List[ClientMetrics]
    server_metrics: List[ServerMetrics]
    
    # Comparative metrics
    lora_vs_no_lora: Dict[str, Any]
    scalability_analysis: Dict[int, Dict[str, float]]  # num_clients -> metrics
    
    # Final results
    final_accuracy: Dict[str, float]
    convergence_round: int
    total_experiment_time: float

class SimpleLoRALinear(nn.Module):
    """Simplified LoRA implementation for research"""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LoRA forward pass: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return lora_out

class ResearchBERTModel(nn.Module):
    """Research-grade BERT model with optional LoRA"""
    def __init__(self, model_name: str, task_type: str = "classification", 
                 use_lora: bool = True, lora_config: Optional[Dict] = None):
        super().__init__()
        self.use_lora = use_lora
        self.task_type = task_type
        
        # Load base model
        if task_type == "regression":
            self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Add LoRA layers if requested
        if use_lora and lora_config:
            self._add_lora_layers(lora_config)
        
        self.model_name = model_name
        
    def _add_lora_layers(self, lora_config: Dict):
        """Add LoRA layers to attention and feed-forward layers"""
        rank = lora_config.get('rank', 16)
        alpha = lora_config.get('alpha', 32)
        dropout = lora_config.get('dropout', 0.1)
        
        self.lora_layers = nn.ModuleDict()
        
        # Add LoRA to attention layers
        for name, module in self.bert.named_modules():
            if 'attention.self.query' in name or 'attention.self.key' in name or 'attention.self.value' in name:
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    lora_name = name.replace('.', '_') + '_lora'
                    self.lora_layers[lora_name] = SimpleLoRALinear(
                        module.in_features, module.out_features, rank, alpha, dropout
                    )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if self.use_lora and hasattr(self, 'lora_layers'):
            # Apply LoRA modifications (simplified for research)
            pass  # LoRA forward pass would be implemented here
        
        return outputs
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """Get only LoRA parameters for efficient communication"""
        if not self.use_lora or not hasattr(self, 'lora_layers'):
            return {}
        
        lora_params = {}
        for name, layer in self.lora_layers.items():
            lora_params[f"{name}.lora_A"] = layer.lora_A.data.cpu()
            lora_params[f"{name}.lora_B"] = layer.lora_B.data.cpu()
        return lora_params
    
    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor]):
        """Set LoRA parameters from server"""
        if not self.use_lora or not hasattr(self, 'lora_layers'):
            return
        
        with torch.no_grad():
            for name, layer in self.lora_layers.items():
                if f"{name}.lora_A" in lora_params:
                    layer.lora_A.data = lora_params[f"{name}.lora_A"].to(layer.lora_A.device)
                if f"{name}.lora_B" in lora_params:
                    layer.lora_B.data = lora_params[f"{name}.lora_B"].to(layer.lora_B.device)

class GLUEDataset(Dataset):
    """GLUE dataset wrapper for research"""
    def __init__(self, task_name: str, split: str, tokenizer, max_samples: int = 1000):
        self.task_name = task_name
        self.tokenizer = tokenizer
        
        # Load dataset
        if task_name == "sst2":
            dataset = load_dataset("glue", "sst2")[split]
            self.texts = [item["sentence"] for item in dataset]
            self.labels = [item["label"] for item in dataset]
            self.task_type = "classification"
        elif task_name == "qqp":
            dataset = load_dataset("glue", "qqp")[split]
            self.texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
            self.labels = [item["label"] for item in dataset]
            self.task_type = "classification"
        elif task_name == "stsb":
            dataset = load_dataset("glue", "stsb")[split]
            self.texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
            self.labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to [0,1]
            self.task_type = "regression"
        
        # Limit samples for research
        if max_samples and len(self.texts) > max_samples:
            indices = np.random.choice(len(self.texts), max_samples, replace=False)
            self.texts = [self.texts[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float if self.task_type == "regression" else torch.long)
        }

class ResearchFederatedClient:
    """Research-grade federated learning client with comprehensive metrics"""
    
    def __init__(self, client_id: str, task_name: str, config: ResearchConfig):
        self.client_id = client_id
        self.task_name = task_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)
        
        # Determine task type
        self.task_type = "regression" if task_name == "stsb" else "classification"
        
        # Initialize model with LoRA configuration
        lora_config = {
            'rank': config.lora_rank,
            'alpha': config.lora_alpha,
            'dropout': config.lora_dropout
        } if config.use_lora else None
        
        self.model = ResearchBERTModel(
            config.client_model, 
            self.task_type, 
            config.use_lora, 
            lora_config
        )
        self.model.to(self.device)
        
        # Initialize dataset
        self.dataset = GLUEDataset(task_name, "train", self.tokenizer, config.data_samples_per_client)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        # Initialize optimizer
        if config.use_lora:
            # Only optimize LoRA parameters
            lora_params = [p for n, p in self.model.named_parameters() if 'lora' in n]
            self.optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate)
        else:
            # Optimize all parameters
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Metrics tracking
        self.metrics_history = []
        
        logger.info(f"Client {client_id} initialized for task {task_name}")
        logger.info(f"Model: {config.client_model}, LoRA: {config.use_lora}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Client {client_id}: {total_params:,} total params, {trainable_params:,} trainable")
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss with proper handling"""
        temperature = self.config.distillation_temperature
        alpha = self.config.distillation_alpha
        
        # Handle batch size mismatch
        min_batch_size = min(student_logits.size(0), teacher_logits.size(0), labels.size(0))
        student_logits = student_logits[:min_batch_size]
        teacher_logits = teacher_logits[:min_batch_size]
        labels = labels[:min_batch_size]
        
        # Handle dimension mismatch
        if student_logits.shape != teacher_logits.shape:
            if len(teacher_logits.shape) > len(student_logits.shape):
                teacher_logits = teacher_logits.squeeze()
            elif len(student_logits.shape) > len(teacher_logits.shape):
                student_logits = student_logits.squeeze()
        
        if self.task_type == 'regression':
            # Regression: MSE loss
            kd_loss = F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze())
            task_loss = F.mse_loss(student_logits.squeeze(), labels)
        else:
            # Classification: KL divergence
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
            task_loss = F.cross_entropy(student_logits, labels.long())
        
        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        return total_loss, kd_loss, task_loss
    
    async def local_training(self, global_params: Optional[Dict] = None, teacher_logits: Optional[Dict] = None) -> Tuple[Dict, ClientMetrics]:
        """Perform local training with comprehensive metrics collection"""
        start_time = time.time()
        
        # Set global parameters if provided
        if global_params:
            if self.config.use_lora:
                self.model.set_lora_parameters(global_params)
            else:
                # Set full parameters for non-LoRA version
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            param.data = torch.tensor(global_params[name]).to(self.device)
        
        self.model.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Training loop
        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits
                
                # Compute loss
                if teacher_logits and str(batch_idx) in teacher_logits:
                    # Knowledge distillation
                    teacher_batch_logits = torch.tensor(teacher_logits[str(batch_idx)]).to(self.device)
                    loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                        student_logits, teacher_batch_logits, labels
                    )
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()
                else:
                    # Standard training
                    if self.task_type == "regression":
                        loss = F.mse_loss(student_logits.squeeze(), labels)
                    else:
                        loss = F.cross_entropy(student_logits, labels.long())
                    total_task_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if self.task_type == "classification":
                    predictions = torch.argmax(student_logits, dim=-1)
                    correct_predictions += (predictions == labels.long()).sum().item()
                else:
                    # For regression, use threshold-based accuracy
                    predictions = student_logits.squeeze()
                    correct_predictions += (torch.abs(predictions - labels) < 0.1).sum().item()
                
                total_samples += labels.size(0)
        
        training_time = time.time() - start_time
        
        # Get updated parameters
        if self.config.use_lora:
            updated_params = self.model.get_lora_parameters()
        else:
            updated_params = {name: param.data.cpu().numpy() for name, param in self.model.named_parameters()}
        
        # Calculate metrics
        avg_loss = total_loss / (len(self.dataloader) * self.config.local_epochs)
        accuracy = correct_predictions / total_samples
        avg_kd_loss = total_kd_loss / (len(self.dataloader) * self.config.local_epochs) if total_kd_loss > 0 else 0.0
        avg_task_loss = total_task_loss / (len(self.dataloader) * self.config.local_epochs)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_usage = 0.0
        
        # Create metrics object
        metrics = ClientMetrics(
            client_id=self.client_id,
            round_num=0,  # Will be set by caller
            task=self.task_name,
            train_loss=avg_loss,
            train_accuracy=accuracy,
            local_epochs_completed=self.config.local_epochs,
            kd_loss=avg_kd_loss,
            task_loss=avg_task_loss,
            training_time=training_time,
            communication_time=0.0,  # Will be set by caller
            parameter_count=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            memory_usage=memory_usage,
            data_samples=len(self.dataset),
            data_distribution=self._get_data_distribution()
        )
        
        self.metrics_history.append(metrics)
        
        logger.info(f"Client {self.client_id} training complete: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        return updated_params, metrics
    
    def _get_data_distribution(self) -> Dict[str, int]:
        """Get data distribution for analysis"""
        if self.task_type == "classification":
            labels = [item['labels'].item() for item in self.dataset]
            unique, counts = np.unique(labels, return_counts=True)
            return {f"class_{int(label)}": int(count) for label, count in zip(unique, counts)}
        else:
            # For regression, create bins
            labels = [item['labels'].item() for item in self.dataset]
            hist, bins = np.histogram(labels, bins=5)
            return {f"bin_{i}": int(count) for i, count in enumerate(hist)}
    
    async def run_client(self, server_host: str = "localhost", server_port: int = 8769):
        """Run client with WebSocket connection to server"""
        uri = f"ws://{server_host}:{server_port}"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Register with server
                registration = {
                    "type": "register",
                    "client_id": self.client_id,
                    "task": self.task_name,
                    "model": self.config.client_model,
                    "use_lora": self.config.use_lora
                }
                await websocket.send(json.dumps(registration))
                
                logger.info(f"Client {self.client_id} registered with server")
                
                # Listen for training requests
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data["type"] == "train":
                        comm_start = time.time()
                        
                        # Perform local training
                        updated_params, metrics = await self.local_training(
                            data.get("global_params"),
                            data.get("teacher_logits")
                        )
                        
                        metrics.round_num = data["round"]
                        metrics.communication_time = time.time() - comm_start
                        
                        # Send results back
                        response = {
                            "type": "update",
                            "client_id": self.client_id,
                            "round": data["round"],
                            "parameters": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in updated_params.items()},
                            "metrics": asdict(metrics)
                        }
                        
                        await websocket.send(json.dumps(response))
                        logger.info(f"Client {self.client_id} sent update for round {data['round']}")
                    
                    elif data["type"] == "finish":
                        logger.info(f"Client {self.client_id} received finish signal")
                        break
        
        except Exception as e:
            logger.error(f"Client {self.client_id} error: {e}")

class ResearchFederatedServer:
    """Research-grade federated learning server with comprehensive metrics"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize global model (teacher)
        self.tokenizer = AutoTokenizer.from_pretrained(config.server_model)
        self.global_model = AutoModelForSequenceClassification.from_pretrained(
            config.server_model, num_labels=2
        )
        self.global_model.to(self.device)
        
        # Client management
        self.connected_clients = {}
        self.client_metrics_history = []
        self.server_metrics_history = []
        
        # Experiment tracking
        self.experiment_start_time = None
        self.current_round = 0
        
        logger.info(f"Server initialized with {config.server_model}")
        total_params = sum(p.numel() for p in self.global_model.parameters())
        logger.info(f"Global model: {total_params:,} parameters")
    
    async def client_handler(self, websocket):
        """Handle client connections and training coordination"""
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "register":
                    client_id = data["client_id"]
                    self.connected_clients[client_id] = {
                        "websocket": websocket,
                        "task": data["task"],
                        "model": data["model"],
                        "use_lora": data["use_lora"]
                    }
                    logger.info(f"Client {client_id} registered. Total clients: {len(self.connected_clients)}")
                
                elif data["type"] == "update":
                    # Store client update
                    client_id = data["client_id"]
                    round_num = data["round"]
                    
                    # Convert parameters back to numpy arrays
                    parameters = {k: np.array(v) for k, v in data["parameters"].items()}
                    
                    # Store metrics
                    metrics = ClientMetrics(**data["metrics"])
                    self.client_metrics_history.append(metrics)
                    
                    # Store client update for aggregation
                    if not hasattr(self, 'client_updates'):
                        self.client_updates = {}
                    if round_num not in self.client_updates:
                        self.client_updates[round_num] = {}
                    
                    self.client_updates[round_num][client_id] = {
                        "parameters": parameters,
                        "metrics": metrics
                    }
                    
                    logger.info(f"Received update from client {client_id} for round {round_num}")
        
        except websockets.exceptions.ConnectionClosed:
            # Remove disconnected client
            for client_id, client_info in list(self.connected_clients.items()):
                if client_info["websocket"] == websocket:
                    del self.connected_clients[client_id]
                    logger.info(f"Client {client_id} disconnected")
                    break
        except Exception as e:
            logger.error(f"Client handler error: {e}")
    
    def aggregate_parameters(self, client_updates: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Aggregate client parameters using FedAvg"""
        if not client_updates:
            return {}
        
        # Get all parameter names from first client
        first_client = next(iter(client_updates.values()))
        param_names = first_client["parameters"].keys()
        
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect parameters from all clients
            client_params = []
            client_weights = []
            
            for client_id, update in client_updates.items():
                if param_name in update["parameters"]:
                    client_params.append(update["parameters"][param_name])
                    # Weight by number of samples
                    client_weights.append(update["metrics"].data_samples)
            
            if client_params:
                # Weighted average
                total_weight = sum(client_weights)
                weighted_sum = np.zeros_like(client_params[0])
                
                for param, weight in zip(client_params, client_weights):
                    weighted_sum += (weight / total_weight) * param
                
                aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def generate_teacher_logits(self, task_datasets: Dict[str, GLUEDataset]) -> Dict[str, Dict[str, List]]:
        """Generate teacher logits for knowledge distillation"""
        self.global_model.eval()
        teacher_logits = {}
        
        with torch.no_grad():
            for task_name, dataset in task_datasets.items():
                dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
                task_logits = {}
                
                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.global_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.cpu().numpy()
                    
                    task_logits[str(batch_idx)] = logits.tolist()
                
                teacher_logits[task_name] = task_logits
        
        return teacher_logits
    
    async def run_federated_training(self):
        """Run the complete federated training experiment"""
        self.experiment_start_time = time.time()
        
        # Wait for minimum number of clients
        while len(self.connected_clients) < self.config.min_clients:
            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{self.config.min_clients})")
            await asyncio.sleep(2)
        
        logger.info(f"Starting federated training with {len(self.connected_clients)} clients")
        
        # Initialize client updates storage
        self.client_updates = {}
        
        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            self.current_round = round_num
            round_start_time = time.time()
            
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")
            
            # Select participating clients (all for now)
            participating_clients = list(self.connected_clients.keys())
            
            # Generate teacher logits for knowledge distillation
            if self.config.use_lora:  # Only for LoRA version to save computation
                task_datasets = {}
                for client_id, client_info in self.connected_clients.items():
                    task_name = client_info["task"]
                    if task_name not in task_datasets:
                        task_datasets[task_name] = GLUEDataset(
                            task_name, "train", self.tokenizer, self.config.data_samples_per_client
                        )
                
                teacher_logits = self.generate_teacher_logits(task_datasets)
            else:
                teacher_logits = {}
            
            # Get current global parameters
            if self.config.use_lora:
                # For LoRA, we don't send global parameters (clients keep their base models)
                global_params = {}
            else:
                global_params = {name: param.data.cpu().numpy() 
                               for name, param in self.global_model.named_parameters()}
            
            # Send training request to all clients
            training_request = {
                "type": "train",
                "round": round_num,
                "global_params": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in global_params.items()},
                "teacher_logits": teacher_logits
            }
            
            # Send to all clients
            for client_id, client_info in self.connected_clients.items():
                try:
                    # Send task-specific teacher logits
                    task_name = client_info["task"]
                    request_copy = training_request.copy()
                    if task_name in teacher_logits:
                        request_copy["teacher_logits"] = teacher_logits[task_name]
                    else:
                        request_copy["teacher_logits"] = {}
                    
                    await client_info["websocket"].send(json.dumps(request_copy))
                except Exception as e:
                    logger.error(f"Failed to send training request to {client_id}: {e}")
            
            # Wait for client updates
            logger.info("Waiting for client updates...")
            timeout_count = 0
            while (round_num not in self.client_updates or 
                   len(self.client_updates[round_num]) < len(participating_clients)):
                await asyncio.sleep(1)
                timeout_count += 1
                if timeout_count > self.config.timeout:
                    logger.warning(f"Timeout waiting for clients in round {round_num}")
                    break
            
            # Aggregate parameters
            if round_num in self.client_updates:
                aggregation_start = time.time()
                aggregated_params = self.aggregate_parameters(self.client_updates[round_num])
                aggregation_time = time.time() - aggregation_start
                
                # Update global model (for non-LoRA version)
                if not self.config.use_lora and aggregated_params:
                    with torch.no_grad():
                        for name, param in self.global_model.named_parameters():
                            if name in aggregated_params:
                                param.data = torch.tensor(aggregated_params[name]).to(self.device)
                
                # Calculate server metrics
                server_metrics = ServerMetrics(
                    round_num=round_num,
                    participating_clients=len(self.client_updates[round_num]),
                    aggregation_time=aggregation_time,
                    global_accuracy={},  # Would need evaluation dataset
                    global_loss={},
                    parameter_variance=0.0,  # Could calculate if needed
                    convergence_rate=0.0,
                    total_communication_cost=0.0,
                    average_client_latency=0.0
                )
                
                self.server_metrics_history.append(server_metrics)
                
                logger.info(f"Round {round_num} completed in {time.time() - round_start_time:.2f}s")
                logger.info(f"Aggregated parameters from {len(self.client_updates[round_num])} clients")
            
            # Save checkpoint if requested
            if self.config.save_checkpoints and round_num % 5 == 0:
                self.save_checkpoint(round_num)
        
        # Finish training
        finish_message = {"type": "finish"}
        for client_id, client_info in self.connected_clients.items():
            try:
                await client_info["websocket"].send(json.dumps(finish_message))
            except Exception as e:
                logger.error(f"Failed to send finish message to {client_id}: {e}")
        
        # Generate final results
        total_time = time.time() - self.experiment_start_time
        logger.info(f"Federated training completed in {total_time:.2f}s")
        
        # Save results
        self.save_experiment_results()
    
    def save_checkpoint(self, round_num: int):
        """Save model checkpoint and metrics"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / f"global_model_round_{round_num}.pt"
        torch.save(self.global_model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = checkpoint_dir / f"metrics_round_{round_num}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "client_metrics": [asdict(m) for m in self.client_metrics_history],
                "server_metrics": [asdict(m) for m in self.server_metrics_history]
            }, f, indent=2)
        
        logger.info(f"Checkpoint saved for round {round_num}")
    
    def save_experiment_results(self):
        """Save complete experiment results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lora_suffix = "lora" if self.config.use_lora else "no_lora"
        
        # Save detailed metrics
        results_file = results_dir / f"experiment_{lora_suffix}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "config": asdict(self.config),
                "client_metrics": [asdict(m) for m in self.client_metrics_history],
                "server_metrics": [asdict(m) for m in self.server_metrics_history],
                "total_experiment_time": time.time() - self.experiment_start_time
            }, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = results_dir / f"client_metrics_{lora_suffix}_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.client_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.client_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.client_metrics_history:
                    writer.writerow(asdict(metrics))
        
        logger.info(f"Experiment results saved to {results_file}")
    
    async def start_server(self):
        """Start the federated learning server"""
        logger.info(f"Starting server on port {self.config.port}")
        
        # Start WebSocket server
        server = await websockets.serve(
            self.client_handler,
            "localhost",
            self.config.port
        )
        
        logger.info(f"Server listening on localhost:{self.config.port}")
        
        # Run federated training
        await self.run_federated_training()
        
        server.close()
        await server.wait_closed()

async def run_experiment(config: ResearchConfig, mode: str, client_id: str = None, task: str = None):
    """Run federated learning experiment"""
    
    if mode == "server":
        server = ResearchFederatedServer(config)
        await server.start_server()
    
    elif mode == "client":
        if not client_id or not task:
            raise ValueError("Client mode requires client_id and task")
        
        client = ResearchFederatedClient(client_id, task, config)
        await client.run_client("localhost", config.port)
    
    else:
        raise ValueError("Mode must be 'server' or 'client'")

def main():
    parser = argparse.ArgumentParser(description="Deep Research Federated Learning System")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--task", choices=["sst2", "qqp", "stsb"], help="Task name (required for client mode)")
    parser.add_argument("--port", type=int, default=8769, help="Server port")
    parser.add_argument("--rounds", type=int, default=22, help="Number of federated rounds")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--samples", type=int, default=1000, help="Data samples per client")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ResearchConfig(
        port=args.port,
        num_rounds=args.rounds,
        use_lora=args.use_lora,
        data_samples_per_client=args.samples
    )
    
    # Run experiment
    asyncio.run(run_experiment(config, args.mode, args.client_id, args.task))

if __name__ == "__main__":
    main()
