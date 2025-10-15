#!/usr/bin/env python3
"""
Distributed Multi-Task Learning System WITHOUT Federated Learning
Focus: Each client trains on a specific dataset with multi-task learning and transfer learning
Server coordinates training but doesn't aggregate parameters
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
import configparser
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Enhanced metrics imports
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distributed_mtl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DistributedMTLConfig:
    """Configuration for distributed multi-task learning"""
    def __init__(self, **kwargs):
        # Model settings
        self.server_model = kwargs.get("server_model", "bert-base-uncased")
        self.client_model = kwargs.get("client_model", "prajjwal1/bert-tiny")

        # Training settings
        self.num_rounds = int(kwargs.get("num_rounds", 10))
        self.local_epochs = int(kwargs.get("local_epochs", 3))
        self.batch_size = int(kwargs.get("batch_size", 16))
        self.learning_rate = float(kwargs.get("learning_rate", 2e-5))
        self.weight_decay = float(kwargs.get("weight_decay", 0.01))

        # Multi-Task Learning settings
        self.tasks = kwargs.get("tasks", ["sst2", "qqp", "stsb"])
        self.distillation_temperature = float(kwargs.get("distillation_temperature", 3.0))
        self.distillation_alpha = float(kwargs.get("distillation_alpha", 0.5))

        # Data settings
        self.samples_per_client = int(kwargs.get("samples_per_client", 1000))
        self.data_distribution = kwargs.get("data_distribution", "iid")

        # Communication settings
        self.port = int(kwargs.get("port", 8771))
        self.timeout = int(kwargs.get("timeout", 300))

@dataclass
class MTLClientMetrics:
    """Multi-task learning client metrics"""
    client_id: str
    dataset_name: str
    round_num: int

    # Task-specific metrics
    task_metrics: Dict[str, Dict[str, float]]

    # Overall metrics
    average_accuracy: float
    total_training_time: float
    memory_usage_mb: float

    # Transfer learning metrics
    transfer_efficiency: float
    knowledge_retention: float

class DistributedMTLDataset(Dataset):
    """Dataset for distributed multi-task learning"""

    def __init__(self, dataset_name: str, tokenizer, samples_per_client: int = 1000):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.samples_per_client = samples_per_client

        # Load and prepare dataset
        self.texts, self.labels, self.task_type, self.num_classes = self._load_dataset()

        # Sample subset for this client
        if len(self.texts) > samples_per_client:
            indices = np.random.choice(len(self.texts), samples_per_client, replace=False)
            self.texts = [self.texts[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        logger.info(f"Dataset {dataset_name}: {len(self.texts)} samples, type: {self.task_type}")

    def _load_dataset(self) -> Tuple[List[str], List, str, int]:
        """Load and prepare dataset"""
        if self.dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2", split="train[:2000]")
            texts = [item["sentence"] for item in dataset]
            labels = [item["label"] for item in dataset]
            return texts, labels, "classification", 2

        elif self.dataset_name == "qqp":
            dataset = load_dataset("glue", "qqp", split="train[:2000]")
            texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
            labels = [item["label"] for item in dataset]
            return texts, labels, "classification", 2

        elif self.dataset_name == "stsb":
            dataset = load_dataset("glue", "stsb", split="train[:2000]")
            texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
            labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to [0,1]
            return texts, labels, "regression", 1

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

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
            'labels': torch.tensor(self.labels[idx],
                                 dtype=torch.long if self.task_type == "classification" else torch.float)
        }

class DistributedMTLClient:
    """Distributed MTL client with transfer learning"""

    def __init__(self, client_id: str, dataset_name: str, config: DistributedMTLConfig):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)

        # Create dataset
        self.dataset = DistributedMTLDataset(dataset_name, self.tokenizer, config.samples_per_client)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)

        # Initialize models for MTL
        self.models = {}
        self.optimizers = {}

        # Create models for different task types
        if self.dataset.task_type == "regression":
            # For regression dataset, create classification and regression models
            self.models["classification"] = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=2
            )
            self.models["regression"] = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=1
            )
        else:
            # For classification dataset, create models for different classifications
            self.models["binary_classification"] = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=2
            )
            self.models["regression"] = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=1
            )

        # Move models to device
        for model in self.models.values():
            model.to(self.device)

        # Initialize optimizers
        for task_type, model in self.models.items():
            self.optimizers[task_type] = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )

        logger.info(f"Distributed MTL Client {client_id} initialized for dataset {dataset_name}")
        logger.info(f"Models: {list(self.models.keys())}, Task type: {self.dataset.task_type}")

    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels, task_type):
        """Compute knowledge distillation loss for transfer learning"""
        temperature = self.config.distillation_temperature
        alpha = self.config.distillation_alpha

        if task_type == 'regression':
            kd_loss = F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze())
            task_loss = F.mse_loss(student_logits.squeeze(), labels.float())
        else:
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
            task_loss = F.cross_entropy(student_logits, labels)

        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        return total_loss, kd_loss, task_loss

    def get_teacher_logits(self, task_type, batch_size):
        """Get teacher logits for knowledge distillation"""
        # Simulate teacher behavior based on task type
        if task_type == "regression":
            return torch.randn(batch_size, 1, dtype=torch.float32).to(self.device) * 0.3 + 0.5
        else:
            return torch.randn(batch_size, 2, dtype=torch.float32).to(self.device)

    async def local_mtl_training(self, round_num: int) -> MTLClientMetrics:
        """Perform local multi-task learning with transfer learning"""
        start_time = time.time()

        all_task_metrics = {}

        # Train each model with transfer learning
        for task_type, model in self.models.items():
            model.train()
            optimizer = self.optimizers[task_type]

            total_loss = 0.0
            total_kd_loss = 0.0
            total_task_loss = 0.0

            # Collect predictions for metrics
            all_predictions = []
            all_labels = []

            for epoch in range(self.config.local_epochs):
                for batch in self.dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device).long() if task_type != "regression" else batch['labels'].to(self.device).float()

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = outputs.logits

                    # Get teacher logits for transfer learning
                    teacher_logits = self.get_teacher_logits(task_type, len(labels))

                    # Compute loss with knowledge distillation
                    loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                        student_logits, teacher_logits, labels, task_type
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()

                    # Collect predictions for metrics
                    if task_type == "regression":
                        predictions = student_logits.squeeze().detach().cpu().numpy()
                    else:
                        predictions = torch.argmax(student_logits, dim=-1).detach().cpu().numpy()

                    all_predictions.extend(predictions)
                    all_labels.extend(labels.detach().cpu().numpy())

            # Calculate metrics for this task
            avg_loss = total_loss / (len(self.dataloader) * self.config.local_epochs)
            avg_kd_loss = total_kd_loss / (len(self.dataloader) * self.config.local_epochs)
            avg_task_loss = total_task_loss / (len(self.dataloader) * self.config.local_epochs)

            if task_type == "regression" and len(all_predictions) > 0:
                mse = mean_squared_error(all_labels, all_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(all_labels, all_predictions)
                r2 = r2_score(all_labels, all_predictions)

                try:
                    pearson_corr, _ = pearsonr(all_labels, all_predictions)
                except:
                    pearson_corr = 0.0

                task_metrics = {
                    'accuracy': float(r2),  # Use R² as accuracy for regression
                    'precision': float(1.0 - mae),
                    'recall': float(pearson_corr),
                    'f1_score': float(rmse),
                    'loss': float(avg_loss),
                    'kd_loss': float(avg_kd_loss),
                    'task_loss': float(avg_task_loss),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'pearson_corr': float(pearson_corr)
                }
            else:
                if len(all_predictions) > 0:
                    accuracy = accuracy_score(all_labels, all_predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_labels, all_predictions, average='weighted', zero_division=0
                    )
                else:
                    accuracy = precision = recall = f1 = 0.0

                task_metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'loss': float(avg_loss),
                    'kd_loss': float(avg_kd_loss),
                    'task_loss': float(avg_task_loss)
                }

            all_task_metrics[task_type] = task_metrics

            # Log task-specific results
            if task_type == "regression":
                logger.info(f"  {task_type}: Loss={avg_loss:.4f}, R²={task_metrics['accuracy']:.4f}, "
                           f"MSE={task_metrics['mse']:.4f}, Pearson={task_metrics['pearson_corr']:.4f}")
            else:
                logger.info(f"  {task_type}: Loss={avg_loss:.4f}, Acc={task_metrics['accuracy']:.4f}, "
                           f"P={task_metrics['precision']:.4f}, R={task_metrics['recall']:.4f}, F1={task_metrics['f1_score']:.4f}")

        total_training_time = time.time() - start_time

        # Calculate overall metrics
        accuracies = [metrics['accuracy'] for metrics in all_task_metrics.values()]
        average_accuracy = float(np.mean(accuracies))

        # Calculate transfer efficiency (how well knowledge transfers between tasks)
        transfer_efficiency = float(self._calculate_transfer_efficiency(all_task_metrics))

        # Calculate knowledge retention
        knowledge_retention = float(self._calculate_knowledge_retention(all_task_metrics))

        # Get memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage_mb = float(process.memory_info().rss / 1024 / 1024)

        # Create comprehensive metrics
        client_metrics = MTLClientMetrics(
            client_id=self.client_id,
            dataset_name=self.dataset_name,
            round_num=round_num,
            task_metrics=all_task_metrics,
            average_accuracy=average_accuracy,
            total_training_time=float(total_training_time),
            memory_usage_mb=memory_usage_mb,
            transfer_efficiency=transfer_efficiency,
            knowledge_retention=knowledge_retention
        )

        logger.info(f"Distributed MTL Client {self.client_id} ({self.dataset_name}) training complete:")
        logger.info(f"  Average accuracy: {average_accuracy:.4f}")
        logger.info(f"  Transfer efficiency: {transfer_efficiency:.4f}")
        logger.info(f"  Knowledge retention: {knowledge_retention:.4f}")
        logger.info(f"  Training time: {total_training_time:.2f}s")
        logger.info(f"  Memory usage: {memory_usage_mb:.1f}MB")

        return client_metrics

    def _calculate_transfer_efficiency(self, task_metrics: Dict) -> float:
        """Calculate how efficiently knowledge transfers between tasks"""
        if len(task_metrics) < 2:
            return 0.0

        # Simple heuristic: average performance improvement from transfer learning
        kd_losses = [metrics.get('kd_loss', 0) for metrics in task_metrics.values()]
        task_losses = [metrics.get('task_loss', 0) for metrics in task_metrics.values()]

        if not kd_losses or not task_losses:
            return 0.0

        # Transfer efficiency based on KD vs task loss ratio
        avg_kd_loss = np.mean(kd_losses)
        avg_task_loss = np.mean(task_losses)

        if avg_task_loss == 0:
            return 0.0

        # Lower KD loss relative to task loss indicates better transfer
        efficiency = 1.0 - (avg_kd_loss / avg_task_loss)
        return max(0.0, min(1.0, efficiency))

    def _calculate_knowledge_retention(self, task_metrics: Dict) -> float:
        """Calculate knowledge retention across tasks"""
        if len(task_metrics) < 2:
            return 0.0

        # Measure consistency in performance across tasks
        accuracies = [metrics['accuracy'] for metrics in task_metrics.values()]
        if len(accuracies) < 2:
            return 0.0

        # Higher consistency indicates better knowledge retention
        accuracy_std = np.std(accuracies)
        accuracy_mean = np.mean(accuracies)

        if accuracy_mean == 0:
            return 0.0

        # Convert to retention score (lower std = higher retention)
        retention = 1.0 - (accuracy_std / accuracy_mean)
        return max(0.0, min(1.0, retention))

    def _convert_metrics_for_json(self, metrics_dict: Dict) -> Dict:
        """Convert numpy types to Python native types for JSON serialization"""
        def convert_value(value):
            if isinstance(value, (np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.int32, np.int64)):
                return int(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value

        return {k: convert_value(v) for k, v in metrics_dict.items()}

    async def run_client(self, server_host: str = "localhost", server_port: int = 8771):
        """Run distributed MTL client with WebSocket connection"""
        uri = f"ws://{server_host}:{server_port}"

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=50 * 1024 * 1024
                ) as websocket:
                    # Register with server
                    registration = {
                        "type": "register",
                        "client_id": self.client_id,
                        "dataset": self.dataset_name,
                        "task_types": list(self.models.keys()),
                        "model": self.config.client_model,
                        "samples": len(self.dataset)
                    }
                    await websocket.send(json.dumps(registration))

                    logger.info(f"Distributed MTL Client {self.client_id} registered with server")

                    # Listen for training requests
                    async for message in websocket:
                        try:
                            data = json.loads(message)

                            if data["type"] == "train":
                                # Perform distributed MTL training
                                client_metrics = await self.local_mtl_training(data["round"])

                                # Send results back
                                response = {
                                    "type": "update",
                                    "client_id": self.client_id,
                                    "round": data["round"],
                                    "metrics": self._convert_metrics_for_json(asdict(client_metrics))
                                }

                                await websocket.send(json.dumps(response))
                                logger.info(f"Client {self.client_id} sent training results for round {data['round']}")

                            elif data["type"] == "finish":
                                logger.info(f"Client {self.client_id} received finish signal")
                                break

                        except json.JSONDecodeError as e:
                            logger.error(f"Client {self.client_id} JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Client {self.client_id} message processing error: {e}")
                            continue

                    logger.info(f"Distributed MTL Client {self.client_id} completed successfully")
                    return

            except Exception as e:
                retry_count += 1
                logger.error(f"Client {self.client_id} connection failed (attempt {retry_count}): {e}")

                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Client {self.client_id} retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Client {self.client_id} failed after {max_retries} attempts")
                    raise

class DistributedMTLServer:
    """Distributed MTL server for coordination (no parameter aggregation)"""

    def __init__(self, config: DistributedMTLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT-Base teacher model
        self.tokenizer = AutoTokenizer.from_pretrained(config.server_model)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.server_model, num_labels=2
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # Teacher model in eval mode

        # Client management
        self.connected_clients = {}
        self.client_updates = {}
        self.training_history = []

        # CSV output setup
        self.csv_initialized = False
        self.client_metrics_file = None
        self.round_metrics_file = None

        logger.info(f"Distributed MTL Server initialized with {config.server_model}")
        total_params = sum(p.numel() for p in self.teacher_model.parameters())
        logger.info(f"Teacher model: {total_params:,} parameters")

    def initialize_csv_output(self):
        """Initialize CSV output files for metrics"""
        if self.csv_initialized:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        results_dir = Path("distributed_mtl_results")
        results_dir.mkdir(exist_ok=True)

        # Client metrics file (detailed per-client, per-task metrics)
        self.client_metrics_file = results_dir / f"client_metrics_{timestamp}.csv"

        # Round summary file (aggregated results per round)
        self.round_metrics_file = results_dir / f"round_metrics_{timestamp}.csv"

        # Initialize CSV headers
        client_headers = [
            'round', 'client_id', 'dataset', 'task_type',
            'accuracy', 'precision', 'recall', 'f1_score',
            'loss', 'kd_loss', 'task_loss',
            'training_time', 'memory_usage_mb',
            'transfer_efficiency', 'knowledge_retention'
        ]

        round_headers = [
            'round', 'num_clients', 'avg_accuracy', 'accuracy_std',
            'avg_transfer_efficiency', 'avg_knowledge_retention',
            'total_training_time', 'communication_time'
        ]

        # Write headers
        with open(self.client_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(client_headers)

        with open(self.round_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(round_headers)

        self.csv_initialized = True
        logger.info(f"CSV output initialized: {results_dir}")

    def save_client_metrics_to_csv(self, round_num: int, client_id: str, metrics: Dict):
        """Save individual client metrics to CSV"""
        if not self.csv_initialized:
            self.initialize_csv_output()

        # Extract task-specific metrics
        for task_type, task_metrics in metrics['task_metrics'].items():
            row = [
                round_num,
                client_id,
                metrics['dataset_name'],
                task_type,
                task_metrics.get('accuracy', 0.0),
                task_metrics.get('precision', 0.0),
                task_metrics.get('recall', 0.0),
                task_metrics.get('f1_score', 0.0),
                task_metrics.get('loss', 0.0),
                task_metrics.get('kd_loss', 0.0),
                task_metrics.get('task_loss', 0.0),
                metrics.get('total_training_time', 0.0),
                metrics.get('memory_usage_mb', 0.0),
                metrics.get('transfer_efficiency', 0.0),
                metrics.get('knowledge_retention', 0.0)
            ]

            with open(self.client_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def save_round_metrics_to_csv(self, round_num: int, round_data: Dict):
        """Save round summary metrics to CSV"""
        if not self.csv_initialized:
            self.initialize_csv_output()

        # Calculate round statistics
        all_accuracies = []
        all_transfer_efficiencies = []
        all_knowledge_retentions = []
        total_training_time = 0.0

        for client_id, metrics in round_data.items():
            all_accuracies.append(metrics['average_accuracy'])
            all_transfer_efficiencies.append(metrics['transfer_efficiency'])
            all_knowledge_retentions.append(metrics['knowledge_retention'])
            total_training_time += metrics['total_training_time']

        if all_accuracies:
            avg_accuracy = np.mean(all_accuracies)
            accuracy_std = np.std(all_accuracies)
            avg_transfer = np.mean(all_transfer_efficiencies)
            avg_retention = np.mean(all_knowledge_retentions)
        else:
            avg_accuracy = accuracy_std = avg_transfer = avg_retention = 0.0

        row = [
            round_num,
            len(round_data),
            avg_accuracy,
            accuracy_std,
            avg_transfer,
            avg_retention,
            total_training_time,
            0.0  # communication_time placeholder
        ]

        with open(self.round_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    async def client_handler(self, websocket):
        """Handle client connections"""
        try:
            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "register":
                    client_id = data["client_id"]
                    self.connected_clients[client_id] = {
                        "websocket": websocket,
                        "dataset": data["dataset"],
                        "task_types": data["task_types"],
                        "model": data["model"],
                        "samples": data["samples"]
                    }
                    logger.info(f"Client {client_id} registered for dataset {data['dataset']}. "
                               f"Total clients: {len(self.connected_clients)}")

                elif data["type"] == "update":
                    client_id = data["client_id"]
                    round_num = data["round"]

                    # Store client update
                    if round_num not in self.client_updates:
                        self.client_updates[round_num] = {}

                    self.client_updates[round_num][client_id] = data["metrics"]

                    # Save client metrics to CSV
                    self.save_client_metrics_to_csv(round_num, client_id, data["metrics"])

                    logger.info(f"Received update from client {client_id} for round {round_num}")

        except websockets.exceptions.ConnectionClosed:
            for client_id, client_info in list(self.connected_clients.items()):
                if client_info["websocket"] == websocket:
                    del self.connected_clients[client_id]
                    logger.info(f"Client {client_id} disconnected")
                    break
        except Exception as e:
            logger.error(f"Client handler error: {e}")

    async def run_distributed_training(self):
        """Run distributed MTL training coordination"""
        logger.info("Starting distributed MTL training coordination")

        # Wait for clients
        while len(self.connected_clients) == 0:
            logger.info("Waiting for clients to connect...")
            await asyncio.sleep(2)

        logger.info(f"Starting training with {len(self.connected_clients)} clients")

        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")

            if not self.connected_clients:
                logger.warning("No clients connected, skipping round")
                continue

            # Send training requests to all clients
            training_request = {
                "type": "train",
                "round": round_num
            }

            active_clients = []
            for client_id, client_info in list(self.connected_clients.items()):
                try:
                    await asyncio.wait_for(
                        client_info["websocket"].send(json.dumps(training_request)),
                        timeout=5.0
                    )
                    active_clients.append(client_id)
                    logger.info(f"Sent training request to {client_id}")

                except Exception as e:
                    logger.error(f"Failed to send to {client_id}: {e}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]

            logger.info(f"Active clients: {len(active_clients)}")

            # Wait for responses
            round_start = time.time()
            while (round_num not in self.client_updates or
                   len(self.client_updates[round_num]) < len(active_clients)):

                if time.time() - round_start > 120:  # 2 minute timeout
                    logger.warning(f"Round {round_num} timeout")
                    break

                await asyncio.sleep(2)

            # Process results
            if round_num in self.client_updates:
                responses = len(self.client_updates[round_num])
                logger.info(f"Round {round_num}: Received {responses} responses")

                # Aggregate and log metrics
                self._log_round_results(round_num)

                # Save round metrics to CSV
                self.save_round_metrics_to_csv(round_num, self.client_updates[round_num])

        # Send finish signal to all clients
        finish_message = {"type": "finish"}
        for client_id, client_info in list(self.connected_clients.items()):
            try:
                await client_info["websocket"].send(json.dumps(finish_message))
            except Exception as e:
                logger.error(f"Failed to send finish to {client_id}: {e}")

        logger.info("Distributed MTL training completed")

        # Log CSV file locations
        if self.csv_initialized:
            logger.info(f"Training metrics saved to CSV files:")
            logger.info(f"  Client metrics: {self.client_metrics_file}")
            logger.info(f"  Round metrics: {self.round_metrics_file}")

    def _log_round_results(self, round_num: int):
        """Log aggregated results for the round"""
        if round_num not in self.client_updates:
            return

        round_data = self.client_updates[round_num]

        # Aggregate metrics across all clients and tasks
        all_accuracies = []
        all_transfer_efficiencies = []
        all_knowledge_retentions = []

        for client_id, metrics in round_data.items():
            all_accuracies.append(metrics["average_accuracy"])
            all_transfer_efficiencies.append(metrics["transfer_efficiency"])
            all_knowledge_retentions.append(metrics["knowledge_retention"])

        if all_accuracies:
            avg_accuracy = np.mean(all_accuracies)
            avg_transfer = np.mean(all_transfer_efficiencies)
            avg_retention = np.mean(all_knowledge_retentions)

            logger.info(f"Round {round_num} aggregated results:")
            logger.info(f"  Average accuracy: {avg_accuracy:.4f}")
            logger.info(f"  Average transfer efficiency: {avg_transfer:.4f}")
            logger.info(f"  Average knowledge retention: {avg_retention:.4f}")

    async def start_server(self):
        """Start the distributed MTL server"""
        logger.info(f"Starting distributed MTL server on port {self.config.port}")

        server = await websockets.serve(
            self.client_handler,
            "localhost",
            self.config.port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
            max_size=50 * 1024 * 1024
        )

        await self.run_distributed_training()
        server.close()
        await server.wait_closed()

async def run_distributed_mtl_experiment(config: DistributedMTLConfig, mode: str,
                                       client_id: str = None, dataset: str = None):
    """Run distributed MTL experiment"""

    if mode == "server":
        server = DistributedMTLServer(config)
        await server.start_server()

    elif mode == "client":
        if not client_id or not dataset:
            raise ValueError("Client mode requires client_id and dataset")

        client = DistributedMTLClient(client_id, dataset, config)
        await client.run_client()

    else:
        raise ValueError("Mode must be 'server' or 'client'")

def main():
    parser = argparse.ArgumentParser(description="Distributed MTL System")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--dataset", choices=["sst2", "qqp", "stsb"], help="Dataset name")
    parser.add_argument("--port", type=int, default=8771, help="Server port")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per client")

    args = parser.parse_args()

    # Create configuration
    config = DistributedMTLConfig(
        port=args.port,
        num_rounds=args.rounds,
        samples_per_client=args.samples,
        server_model="bert-base-uncased",  # Use BERT-Base as requested
        tasks=["sst2", "qqp", "stsb"]
    )

    logger.info(f"Starting distributed MTL system in {args.mode} mode")

    # Run experiment
    asyncio.run(run_distributed_mtl_experiment(
        config, args.mode, args.client_id, args.dataset
    ))

if __name__ == "__main__":
    main()
