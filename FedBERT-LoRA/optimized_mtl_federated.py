#!/usr/bin/env python3
"""
Optimized MTL Federated Learning System
Fixed connectivity issues and improved performance
"""

import asyncio
import websockets
import json
import logging
import argparse
import configparser
import time
import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import psutil
import os
from pathlib import Path
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch
import numpy as np

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_mtl_federated.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedConfig:
    def __init__(self, **kwargs):
        # Model settings
        self.server_model = kwargs.get("server_model", "prajjwal1/bert-tiny")
        self.client_model = kwargs.get("client_model", "prajjwal1/bert-tiny")

        # Training settings optimized for regression
        self.num_rounds = int(kwargs.get("num_rounds", 3))  # Reduced for faster testing
        self.min_clients = int(kwargs.get("min_clients", 2))  # Wait for at least 2 clients
        self.max_clients = int(kwargs.get("max_clients", 3))
        self.local_epochs = int(kwargs.get("local_epochs", 1))  # Reduced for faster training
        self.batch_size = int(kwargs.get("batch_size", 16))  # Increased for faster processing
        self.learning_rate = float(kwargs.get("learning_rate", 2e-4))  # Slightly higher LR for faster convergence
        self.weight_decay = float(kwargs.get("weight_decay", 0.01))

        # Knowledge Distillation settings
        self.distillation_temperature = float(kwargs.get("distillation_temperature", 3.0))
        self.distillation_alpha = float(kwargs.get("distillation_alpha", 0.5))

        # Data settings
        self.samples_per_client = int(kwargs.get("samples_per_client", 500))  # Reduced for testing
        self.max_samples_per_client = int(kwargs.get("max_samples_per_client", 500))
        self.data_samples_per_client = min(self.samples_per_client, self.max_samples_per_client)
        self.data_distribution = kwargs.get("data_distribution", "non_iid")
        self.non_iid_alpha = float(kwargs.get("non_iid_alpha", 0.5))
        self.oversample_minority = bool(kwargs.get("oversample_minority", True))
        self.normalize_weights = bool(kwargs.get("normalize_weights", True))

        # Communication settings with optimized timeouts
        self.port = int(kwargs.get("port", 8771))
        self.timeout = int(kwargs.get("timeout", 300))  # Increased to 5 minutes for client connection
        self.websocket_timeout = int(kwargs.get("websocket_timeout", 60))  # Increased for message handling
        self.retry_attempts = int(kwargs.get("retry_attempts", 3))  # Reduced retry attempts for faster failure

@dataclass
class ClientParticipationMetrics:
    """Enhanced client participation metrics"""
    client_id: str
    round_num: int
    participated: bool
    participation_rate: float
    consecutive_participations: int
    total_participations: int
    data_samples: int
    data_distribution: Dict[str, int]
    data_heterogeneity_score: float
    local_accuracy: float
    local_precision: float
    local_recall: float
    local_f1_score: float
    local_loss: float
    contribution_weight: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix_flat: List[int]
    upload_time: float
    download_time: float
    parameter_size_bytes: int

class OptimizedMultiTaskFederatedClient:
    """Optimized MTL federated client with improved connectivity"""

    def __init__(self, client_id: str, tasks: List[str], config: OptimizedConfig, total_clients: int):
        self.client_id = client_id
        self.tasks = tasks
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_clients = total_clients

        # Initialize tokenizer (shared across tasks)
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)

        # Initialize student models for each task
        self.student_models = {}
        self.student_optimizers = {}

        for task_name in tasks:
            # Determine task type and number of classes
            if task_name == "stsb":
                task_type = "regression"
                num_labels = 1
            else:
                task_type = "classification"
                num_labels = 2

            # Initialize student model
            model = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=num_labels
            )
            model.to(self.device)

            # Initialize optimizer with better settings
            optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

            self.student_models[task_name] = model
            self.student_optimizers[task_name] = optimizer

        # Initialize dataset with optimized settings
        client_num = 0
        if '_' in client_id:
            try:
                client_num = int(client_id.split('_')[-1])
            except ValueError:
                # If the part after underscore is not a number, use hash of client_id
                client_num = hash(client_id) % 1000
        self.dataset = MultiTaskFederatedDataset(
            tasks, self.tokenizer, client_num, total_clients,
            config.data_samples_per_client, config.data_distribution, config.non_iid_alpha
        )

        # Initialize teacher model
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.server_model, num_labels=1
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        logger.info(f"Optimized MTL Client {client_id} initialized for {len(tasks)} tasks")

    async def run_client(self, server_host: str = "localhost", server_port: int = 8771):
        """Run optimized MTL federated client with improved connection handling"""
        uri = f"ws://{server_host}:{server_port}"

        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Client {self.client_id} connecting (attempt {attempt + 1}/{self.config.retry_attempts})")

                async with websockets.connect(
                    uri,
                    ping_interval=30,  # Increased ping interval for longer training
                    ping_timeout=15,   # Increased ping timeout
                    close_timeout=15,  # Increased close timeout
                    max_size=50 * 1024 * 1024,  # 50MB max message size
                    compression=None
                ) as websocket:

                    # Register with server
                    registration = {
                        "type": "register",
                        "client_id": self.client_id,
                        "tasks": self.tasks,
                        "model": self.config.client_model,
                        "data_distributions": {task: self.dataset.task_data[task]['distribution'] for task in self.tasks},
                        "total_samples": sum(len(self.dataset.task_data[task]['texts']) for task in self.tasks)
                    }

                    await websocket.send(json.dumps(registration))
                    logger.info(f"Client {self.client_id} registered successfully")

                    # Listen for messages with improved error handling and longer timeout
                    while True:
                        try:
                            # Increase timeout for multiple rounds and longer training
                            message = await asyncio.wait_for(websocket.recv(), timeout=self.config.websocket_timeout * 6)
                            data = json.loads(message)

                            if data["type"] == "train":
                                # Perform training
                                updated_params, training_metrics, task_specific_metrics = await self.local_training(
                                    data.get("global_params"), data.get("teacher_logits"), data["round"]
                                )

                                # Send results back with retry logic
                                response = {
                                    "type": "update",
                                    "client_id": self.client_id,
                                    "round": data["round"],
                                    "parameters": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                 for k, v in updated_params.items()},
                                    "training_metrics": asdict(training_metrics),
                                    "task_specific_metrics": task_specific_metrics
                                }

                                # Try to send results with retry
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        await websocket.send(json.dumps(response))
                                        logger.info(f"Client {self.client_id} sent training results for round {data['round']}")
                                        break
                                    except Exception as e:
                                        if attempt < max_retries - 1:
                                            logger.warning(f"Client {self.client_id} failed to send results (attempt {attempt + 1}), retrying...")
                                            await asyncio.sleep(1)
                                        else:
                                            logger.error(f"Client {self.client_id} failed to send results after {max_retries} attempts: {e}")
                                            raise

                            elif data["type"] == "finish":
                                logger.info(f"Client {self.client_id} received finish signal")
                                break

                        except asyncio.TimeoutError:
                            logger.warning(f"Client {self.client_id} message timeout, continuing...")
                            continue
                        except json.JSONDecodeError as e:
                            logger.error(f"Client {self.client_id} JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Client {self.client_id} message processing error: {e}")
                            break

                    logger.info(f"Client {self.client_id} completed successfully")
                    return

            except Exception as e:
                logger.error(f"Client {self.client_id} connection failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    logger.info(f"Client {self.client_id} retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Client {self.client_id} failed after {self.config.retry_attempts} attempts")
                    raise

    async def local_training(self, global_params: Optional[Dict] = None, teacher_logits: Optional[Dict] = None, round_num: int = 0) -> Tuple[Dict, ClientParticipationMetrics, Dict]:
        """Perform optimized local training"""
        start_time = time.time()

        # Training loop (simplified for optimization)
        all_metrics = {}

        for task_name in self.tasks:
            model = self.student_models[task_name]
            optimizer = self.student_optimizers[task_name]
            dataloader = self.dataset.get_task_dataloader(task_name, self.config.batch_size)

            model.train()
            total_loss = 0.0

            # Collect predictions for metrics
            all_predictions = []
            all_labels = []

            for epoch in range(self.config.local_epochs):
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Simple loss calculation (optimized)
                    if self.dataset.task_types[task_name] == "regression":
                        loss = F.mse_loss(logits.squeeze(), labels)
                        predictions = logits.squeeze().detach().cpu().numpy()
                    else:
                        loss = F.cross_entropy(logits, labels.long())
                        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.detach().cpu().numpy())

            # Calculate metrics (simplified)
            avg_loss = total_loss / (len(dataloader) * self.config.local_epochs)

            task_info = self.dataset.get_task_info(task_name)

            if task_info['task_type'] == "classification" and len(all_predictions) > 0:
                accuracy = accuracy_score(all_labels, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_predictions, average='weighted', zero_division=0
                )
            else:
                # For regression, use simplified metrics
                if len(all_predictions) > 0:
                    mse = mean_squared_error(all_labels, all_predictions)
                    accuracy = max(0, 1 - mse)  # Convert MSE to pseudo-accuracy
                    precision = recall = f1 = accuracy
                else:
                    accuracy = precision = recall = f1 = 0.0

            # Calculate validation metrics if validation data is available
            val_accuracy = val_precision = val_recall = val_f1 = val_mse = 0.0
            if 'val_texts' in task_info and len(task_info['val_texts']) > 0:
                val_dataloader = self.dataset.get_task_dataloader(task_name, self.config.batch_size, validation=True)
                model.eval()

                val_predictions = []
                val_labels_list = []

                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                        if task_info['task_type'] == "regression":
                            predictions = logits.squeeze().detach().cpu().numpy()
                        else:
                            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()

                        val_predictions.extend(predictions)
                        val_labels_list.extend(labels.detach().cpu().numpy())

                if task_info['task_type'] == "classification" and len(val_predictions) > 0:
                    val_accuracy = accuracy_score(val_labels_list, val_predictions)
                    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                        val_labels_list, val_predictions, average='weighted', zero_division=0
                    )
                elif task_info['task_type'] == "regression" and len(val_predictions) > 0:
                    val_mse = mean_squared_error(val_labels_list, val_predictions)
                    val_accuracy = max(0, 1 - val_mse)
                    val_precision = val_recall = val_f1 = val_accuracy

            all_metrics[task_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': avg_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1_score': val_f1,
                'val_loss': val_mse if task_info['task_type'] == "regression" else 0.0,
                'data_samples': len(task_info['texts']),
                'validation_samples': len(task_info.get('val_texts', []))
            }

        # Create training metrics with detailed task-specific metrics
        classification_accuracies = []
        classification_precisions = []
        classification_recalls = []
        classification_f1s = []
        regression_accuracies = []
        regression_mses = []

        # Separate metrics by task type
        for task_name, metrics in all_metrics.items():
            if self.dataset.task_types[task_name] == "classification":
                classification_accuracies.append(metrics['accuracy'])
                classification_precisions.append(metrics['precision'])
                classification_recalls.append(metrics['recall'])
                classification_f1s.append(metrics['f1_score'])
            elif self.dataset.task_types[task_name] == "regression":
                regression_accuracies.append(metrics['accuracy'])
                regression_mses.append(metrics['loss'])  # Using training loss as MSE for regression

        # Calculate averages for each task type (using validation metrics where available)
        avg_classification_accuracy = np.mean(classification_accuracies) if classification_accuracies else 0.0
        avg_classification_precision = np.mean(classification_precisions) if classification_precisions else 0.0
        avg_classification_recall = np.mean(classification_recalls) if classification_recalls else 0.0
        avg_classification_f1 = np.mean(classification_f1s) if classification_f1s else 0.0
        avg_regression_accuracy = np.mean(regression_accuracies) if regression_accuracies else 0.0
        avg_regression_mse = np.mean(regression_mses) if regression_mses else 0.0

        # Use validation metrics for overall accuracy if available
        overall_accuracies = []
        for task_name, metrics in all_metrics.items():
            if 'val_accuracy' in metrics and metrics['val_accuracy'] > 0:
                overall_accuracies.append(metrics['val_accuracy'])
            else:
                overall_accuracies.append(metrics['accuracy'])

        final_overall_accuracy = np.mean(overall_accuracies) if overall_accuracies else 0.0

        training_metrics = ClientParticipationMetrics(
            client_id=self.client_id,
            round_num=round_num,
            participated=True,
            participation_rate=1.0,
            consecutive_participations=1,
            total_participations=1,
            data_samples=sum(len(self.dataset.task_data[task]['texts']) for task in self.tasks),
            data_distribution={task: self.dataset.task_data[task]['distribution'] for task in self.tasks},
            data_heterogeneity_score=0.0,
            local_accuracy=final_overall_accuracy,
            local_precision=np.mean([metrics['precision'] for metrics in all_metrics.values()]),
            local_recall=np.mean([metrics['recall'] for metrics in all_metrics.values()]),
            local_f1_score=np.mean([metrics['f1_score'] for metrics in all_metrics.values()]),
            local_loss=np.mean([metrics['loss'] for metrics in all_metrics.values()]),
            contribution_weight=sum(len(self.dataset.task_data[task]['texts']) for task in self.tasks),
            per_class_precision={},
            per_class_recall={},
            per_class_f1={},
            confusion_matrix_flat=[],
            upload_time=0.0,
            download_time=0.0,
            parameter_size_bytes=1000  # Placeholder
        )

        # Get updated parameters (minimal for optimization)
        updated_params = {}
        param_count = 0
        for task_name in self.tasks:
            model = self.student_models[task_name]
            for name, param in model.named_parameters():
                if param_count < 1:  # Reduced to 1 parameter per task for optimization
                    param_key = f"{task_name}_{name}"
                    # Convert to smaller data type and reduce precision
                    param_data = param.data.cpu().numpy().astype(np.float32)
                    updated_params[param_key] = param_data
                    param_count += 1

        total_training_time = time.time() - start_time
        logger.info(f"Client {self.client_id} training completed in {total_training_time:.2f}s")

        # Return updated parameters, training metrics, and task-specific metrics
        return updated_params, training_metrics, {
            'classification_accuracy': avg_classification_accuracy,
            'regression_accuracy': avg_regression_accuracy,
            'classification_precision': avg_classification_precision,
            'classification_recall': avg_classification_recall,
            'classification_f1': avg_classification_f1,
            'regression_mse': avg_regression_mse,
            'validation_classification_accuracy': np.mean([metrics.get('val_accuracy', 0) for task_name, metrics in all_metrics.items() if task_name in ['sst2', 'qqp']]),
            'validation_regression_accuracy': np.mean([metrics.get('val_accuracy', 0) for task_name, metrics in all_metrics.items() if task_name == 'stsb'])
        }

class OptimizedMTLFederatedServer:
    """Optimized MTL Federated server with improved connectivity"""

    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize global model
        self.tokenizer = AutoTokenizer.from_pretrained(config.server_model)
        self.global_model = AutoModelForSequenceClassification.from_pretrained(
            config.server_model, num_labels=1
        )
        self.global_model.to(self.device)

        # Client management with connection monitoring
        self.connected_clients = {}
        self.client_updates = {}
        self.metrics_history = []

        # Initialize CSV file for results in organized folder structure
        results_dir = "federated_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(results_dir, f"federated_results_{timestamp}.csv")
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Create README for results folder
        self._create_results_readme(results_dir)

        # Write CSV header
        self.csv_writer.writerow([
            'round', 'responses_received', 'avg_accuracy', 'client_accuracies',
            'classification_accuracy', 'regression_accuracy', 'classification_precision',
            'classification_recall', 'classification_f1', 'regression_mse',
            'validation_classification_accuracy', 'validation_regression_accuracy',
            'total_clients', 'active_clients', 'training_time', 'timestamp'
        ])

        logger.info(f"Optimized MTL Server initialized")

    async def client_handler(self, websocket):
        """Handle client connections with improved error handling"""
        client_id = None
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data["type"] == "register":
                        client_id = data["client_id"]
                        self.connected_clients[client_id] = {
                            "websocket": websocket,
                            "tasks": data["tasks"],
                            "model": data["model"],
                            "data_distributions": data["data_distributions"],
                            "total_samples": data["total_samples"],
                            "last_seen": time.time()
                        }
                        logger.info(f"Client {client_id} registered. Total clients: {len(self.connected_clients)}")

                    elif data["type"] == "update":
                        client_id = data["client_id"]
                        round_num = data["round"]

                        if client_id in self.connected_clients:
                            self.connected_clients[client_id]["last_seen"] = time.time()

                        # Store client update
                        if round_num not in self.client_updates:
                            self.client_updates[round_num] = {}

                        self.client_updates[round_num][client_id] = data
                        logger.info(f"Received update from client {client_id} for round {round_num}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            if client_id and client_id in self.connected_clients:
                del self.connected_clients[client_id]
                logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            if client_id and client_id in self.connected_clients:
                del self.connected_clients[client_id]

    async def run_federated_training(self):
        """Run optimized federated training"""
        logger.info("Starting optimized federated training")

        # Wait for clients with improved monitoring
        start_time = time.time()
        while len(self.connected_clients) < self.config.min_clients:
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout:
                logger.warning(f"Timeout waiting for clients after {elapsed:.0f}s")
                break

            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{self.config.min_clients})")
            await asyncio.sleep(2)

        # Additional wait for more clients if we have the minimum
        if len(self.connected_clients) >= self.config.min_clients:
            logger.info(f"Have minimum clients ({len(self.connected_clients)}), waiting a bit longer for more clients...")
            additional_wait_start = time.time()
            previous_client_count = len(self.connected_clients)
            while time.time() - additional_wait_start < 10:  # Wait up to 10 more seconds
                await asyncio.sleep(1)
                current_clients = len(self.connected_clients)
                if current_clients > previous_client_count:
                    logger.info(f"More clients joined! Now have {current_clients} clients")
                    previous_client_count = current_clients
                    additional_wait_start = time.time()  # Reset timer for more waiting
                elif time.time() - additional_wait_start >= 10:
                    logger.info(f"Finished waiting for additional clients. Final count: {current_clients}")
                    break

        logger.info(f"Starting training with {len(self.connected_clients)} clients")

        # Training rounds with optimized timing
        for round_num in range(1, self.config.num_rounds + 1):
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")

            if not self.connected_clients:
                logger.warning("No clients connected, skipping round")
                continue

            round_start = time.time()  # Add this line to define round_start

            # Send training requests with improved error handling
            training_request = {
                "type": "train",
                "round": round_num,
                "global_params": {},
                "teacher_logits": {}
            }

            active_clients = []
            for client_id, client_info in list(self.connected_clients.items()):
                try:
                    # Check if client is still responsive
                    if time.time() - client_info["last_seen"] > 30:  # 30 second timeout
                        logger.warning(f"Client {client_id} not responding, removing")
                        del self.connected_clients[client_id]
                        continue

                    await asyncio.wait_for(
                        client_info["websocket"].send(json.dumps(training_request)),
                        timeout=5.0  # Reduced timeout for faster failure detection
                    )
                    active_clients.append(client_id)
                    logger.info(f"Sent training request to {client_id}")

                except Exception as e:
                    logger.error(f"Failed to send to {client_id}: {e}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]

            logger.info(f"Active clients: {len(active_clients)}")

            # Wait for client updates
            logger.info("Waiting for client updates...")
            timeout_count = 0
            expected_clients = len(active_clients)

            while (round_num not in self.client_updates or
                   len(self.client_updates[round_num]) < expected_clients):

                if time.time() - round_start > 60:  # Reduced timeout per round (60 seconds)
                    logger.warning(f"Round {round_num} timeout")
                    break

                await asyncio.sleep(2)  # Check every 2 seconds instead of 1

                # Log progress every 15 seconds
                if timeout_count % 8 == 0:  # Every 15 seconds (8 * 2s)
                    received = len(self.client_updates.get(round_num, {}))
                    logger.info(f"Round {round_num}: Received {received}/{expected_clients} client updates (timeout in {60 - timeout_count*2}s)")
                timeout_count += 1

            # Calculate and log average metrics
            responses = len(self.client_updates.get(round_num, {}))
            if responses > 0:
                accuracies = []
                classification_accuracies = []
                regression_accuracies = []
                classification_precisions = []
                classification_recalls = []
                classification_f1s = []
                regression_mses = []
                validation_classification_accuracies = []
                validation_regression_accuracies = []

                client_metrics = {}
                task_type_metrics = {}

                for client_id, update in self.client_updates[round_num].items():
                    metrics = update.get("training_metrics", {})
                    if "local_accuracy" in metrics:
                        client_acc = metrics["local_accuracy"]
                        accuracies.append(client_acc)
                        client_metrics[client_id] = client_acc

                    # Collect detailed task-specific metrics from task_specific_metrics dict
                    task_metrics = update.get("task_specific_metrics", {})
                    if task_metrics:
                        logger.info(f"DEBUG: Client {client_id} task metrics: {task_metrics}")  # Debug line
                        if 'classification_accuracy' in task_metrics:
                            classification_accuracies.append(task_metrics['classification_accuracy'])
                        if 'regression_accuracy' in task_metrics:
                            regression_accuracies.append(task_metrics['regression_accuracy'])
                        if 'classification_precision' in task_metrics:
                            classification_precisions.append(task_metrics['classification_precision'])
                        if 'classification_recall' in task_metrics:
                            classification_recalls.append(task_metrics['classification_recall'])
                        if 'classification_f1' in task_metrics:
                            classification_f1s.append(task_metrics['classification_f1'])
                        if 'regression_mse' in task_metrics:
                            regression_mses.append(task_metrics['regression_mse'])
                        if 'validation_classification_accuracy' in task_metrics:
                            validation_classification_accuracies.append(task_metrics['validation_classification_accuracy'])
                        if 'validation_regression_accuracy' in task_metrics:
                            validation_regression_accuracies.append(task_metrics['validation_regression_accuracy'])

                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    logger.info(f"Round {round_num} average accuracy: {avg_accuracy:.4f}")

                    # Calculate aggregated metrics across all clients
                    final_classification_accuracy = np.mean(classification_accuracies) if classification_accuracies else 0.0
                    final_regression_accuracy = np.mean(regression_accuracies) if regression_accuracies else 0.0
                    final_classification_precision = np.mean(classification_precisions) if classification_precisions else 0.0
                    final_classification_recall = np.mean(classification_recalls) if classification_recalls else 0.0
                    final_classification_f1 = np.mean(classification_f1s) if classification_f1s else 0.0
                    final_regression_mse = np.mean(regression_mses) if regression_mses else 0.0
                    final_validation_classification_accuracy = np.mean([metrics.get('val_accuracy', 0) for task_name, metrics in all_metrics.items() if task_name in ['sst2', 'qqp']]) if any(task_name in all_metrics for task_name in ['sst2', 'qqp']) else 0.0
                    final_validation_regression_accuracy = np.mean([metrics.get('val_accuracy', 0) for task_name, metrics in all_metrics.items() if task_name == 'stsb']) if 'stsb' in all_metrics else 0.0

                    # Save results to CSV with detailed metrics
                    training_time = time.time() - round_start if 'round_start' in locals() else 0
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    try:
                        self.csv_writer.writerow([
                            round_num,
                            responses,
                            f"{avg_accuracy:.4f}",
                            str(client_metrics),
                            f"{final_classification_accuracy:.4f}",
                            f"{final_regression_accuracy:.4f}",
                            f"{final_classification_precision:.4f}",
                            f"{final_classification_recall:.4f}",
                            f"{final_classification_f1:.4f}",
                            f"{final_regression_mse:.4f}",
                            f"{final_validation_classification_accuracy:.4f}",
                            f"{final_validation_regression_accuracy:.4f}",
                            len(self.connected_clients),
                            expected_clients,
                            f"{training_time:.2f}",
                            current_timestamp
                        ])
                        self.csv_file.flush()  # Force write to disk
                    except Exception as e:
                        logger.error(f"Error writing to CSV file: {e}")

            # Clean up old connections
            current_time = time.time()
            to_remove = []
            for client_id, client_info in self.connected_clients.items():
                if current_time - client_info["last_seen"] > 60:  # 60 second cleanup
                    to_remove.append(client_id)

            for client_id in to_remove:
                del self.connected_clients[client_id]
                logger.info(f"Cleaned up stale client {client_id}")

        logger.info("Federated training completed")

        # Close CSV file
        if hasattr(self, 'csv_file'):
            try:
                self.csv_file.close()
                logger.info(f"Results saved to {self.csv_filename}")

                # Create summary analysis
                self._create_summary_analysis()

            except Exception as e:
                logger.error(f"Error closing CSV file: {e}")

    def _create_summary_analysis(self):
        """Create summary analysis of the federated training results"""
        try:
            # Read the CSV file to analyze results
            with open(self.csv_filename, 'r') as f:
                reader = csv.DictReader(f)
                data = list(reader)

            if data:
                # Calculate final statistics
                final_round = max(int(row['round']) for row in data)
                final_accuracy = float(data[-1]['avg_accuracy'])
                avg_training_time = np.mean([float(row['training_time']) for row in data])

                # Create summary file
                summary_file = os.path.join(os.path.dirname(self.csv_filename), "training_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write("Federated Learning Training Summary\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Total Rounds: {final_round}\n")
                    f.write(f"Final Average Accuracy: {final_accuracy:.4f}\n")
                    f.write(f"Average Training Time per Round: {avg_training_time:.2f}s\n")
                    f.write(f"Results File: {self.csv_filename}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                logger.info(f"Summary analysis saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error creating summary analysis: {e}")

    def _create_results_readme(self, results_dir):
        """Create README file explaining the results folder structure"""
        readme_path = os.path.join(results_dir, "README.md")
        try:
            with open(readme_path, 'w') as f:
                f.write("# Federated Learning Results\n\n")
                f.write("This folder contains results from federated learning experiments.\n\n")
                f.write("## Files Description:\n\n")
                f.write("### CSV Files (`federated_results_YYYYMMDD_HHMMSS.csv`)\n")
                f.write("- **round**: Training round number\n")
                f.write("- **responses_received**: Number of client updates received\n")
                f.write("- **avg_accuracy**: Overall average accuracy across all tasks\n")
                f.write("- **client_accuracies**: Individual client accuracy scores (JSON format)\n")
                f.write("- **classification_accuracy**: Average accuracy for classification tasks (SST2, QQP)\n")
                f.write("- **regression_accuracy**: Average accuracy for regression tasks (STSB)\n")
                f.write("- **classification_precision**: Average precision for classification tasks\n")
                f.write("- **classification_recall**: Average recall for classification tasks\n")
                f.write("- **classification_f1**: Average F1-score for classification tasks\n")
                f.write("- **regression_mse**: Mean Squared Error for regression tasks\n")
                f.write("- **validation_classification_accuracy**: Average validation accuracy for classification tasks (SST2, QQP)\n")
                f.write("- **validation_regression_accuracy**: Average validation accuracy for regression tasks (STSB)\n")
                f.write("- **total_clients**: Total number of connected clients\n")
                f.write("- **active_clients**: Number of clients active for this round\n")
                f.write("- **training_time**: Time taken for the round (seconds)\n")
                f.write("- **timestamp**: Date/time when results were recorded\n\n")
                f.write("### Summary File (`training_summary.txt`)\n")
                f.write("- Contains overall training statistics and final results\n")
                f.write("- Generated automatically after training completion\n\n")
                f.write("## Usage:\n")
                f.write("- Import CSV files into Excel, Google Sheets, or pandas for analysis\n")
                f.write("- Use timestamp in filename to track different experiments\n")
                f.write("- Compare results across different federated learning configurations\n")

            logger.info(f"Results README created at {readme_path}")

        except Exception as e:
            logger.error(f"Error creating results README: {e}")

    def __del__(self):
        """Destructor to ensure CSV file is closed"""
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            try:
                self.csv_file.close()
            except:
                pass

    async def start_server(self):
        """Start optimized server"""
        logger.info(f"Starting optimized server on port {self.config.port}")

        server = await websockets.serve(
            self.client_handler,
            "localhost",
            self.config.port,
            ping_interval=30,  # Increased ping interval for longer training
            ping_timeout=15,   # Increased ping timeout
            close_timeout=15,  # Increased close timeout
            max_size=100 * 1024 * 1024  # Increased to 100MB
        )

        await self.run_federated_training()
        server.close()
        await server.wait_closed()

# Simplified dataset class for optimization
class MultiTaskFederatedDataset:
    def __init__(self, tasks: List[str], tokenizer, client_id: int, total_clients: int,
                 samples_per_client: int, distribution_type: str = "non_iid", alpha: float = 0.5):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.client_id = client_id
        self.total_clients = total_clients
        self.samples_per_client = samples_per_client
        self.distribution_type = distribution_type

        # Load datasets for all tasks
        self.task_datasets = {}
        self.task_types = {}  # Add task_types attribute
        self.task_num_classes = {}
        self.task_data = {}  # Add task_data attribute

        for task_name in tasks:
            if task_name == "sst2":
                # Load full SST-2 dataset
                dataset = load_dataset("glue", "sst2", split="train")
                texts = [item["sentence"] for item in dataset]
                labels = [item["label"] for item in dataset]
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "classification"
                self.task_num_classes[task_name] = 2

                # Create train/validation split (80/20)
                total_samples = len(texts)
                train_size = int(0.8 * total_samples)
                val_size = total_samples - train_size

                # Shuffle indices for random split
                indices = list(range(total_samples))
                random.shuffle(indices)

                train_indices = indices[:train_size]
                val_indices = indices[train_size:]

                train_texts = [texts[i] for i in train_indices]
                train_labels = [labels[i] for i in train_indices]
                val_texts = [texts[i] for i in val_indices]
                val_labels = [labels[i] for i in val_indices]

                logger.info(f"Task {task_name}: Total={total_samples}, Train={len(train_texts)}, Validation={len(val_texts)}")

                self.task_data[task_name] = {
                    'texts': train_texts,
                    'labels': train_labels,
                    'val_texts': val_texts,
                    'val_labels': val_labels,
                    'task_type': self.task_types[task_name],
                    'distribution': {'data': len(train_labels), 'validation': len(val_labels)}
                }

            elif task_name == "qqp":
                # Load full QQP dataset
                dataset = load_dataset("glue", "qqp", split="train")
                texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
                labels = [item["label"] for item in dataset]
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "classification"
                self.task_num_classes[task_name] = 2

                # Create train/validation split (80/20)
                total_samples = len(texts)
                train_size = int(0.8 * total_samples)
                val_size = total_samples - train_size

                # Shuffle indices for random split
                indices = list(range(total_samples))
                random.shuffle(indices)

                train_indices = indices[:train_size]
                val_indices = indices[train_size:]

                train_texts = [texts[i] for i in train_indices]
                train_labels = [labels[i] for i in train_indices]
                val_texts = [texts[i] for i in val_indices]
                val_labels = [labels[i] for i in val_indices]

                logger.info(f"Task {task_name}: Total={total_samples}, Train={len(train_texts)}, Validation={len(val_texts)}")

                self.task_data[task_name] = {
                    'texts': train_texts,
                    'labels': train_labels,
                    'val_texts': val_texts,
                    'val_labels': val_labels,
                    'task_type': self.task_types[task_name],
                    'distribution': {'data': len(train_labels), 'validation': len(val_labels)}
                }

            elif task_name == "stsb":
                # Load full STSB dataset
                dataset = load_dataset("glue", "stsb", split="train")
                texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
                labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to 0-1
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "regression"
                self.task_num_classes[task_name] = 1

                # Create train/validation split (80/20)
                total_samples = len(texts)
                train_size = int(0.8 * total_samples)
                val_size = total_samples - train_size

                # Shuffle indices for random split
                indices = list(range(total_samples))
                random.shuffle(indices)

                train_indices = indices[:train_size]
                val_indices = indices[train_size:]

                train_texts = [texts[i] for i in train_indices]
                train_labels = [labels[i] for i in train_indices]
                val_texts = [texts[i] for i in val_indices]
                val_labels = [labels[i] for i in val_indices]

                logger.info(f"Task {task_name}: Total={total_samples}, Train={len(train_texts)}, Validation={len(val_texts)}")

                self.task_data[task_name] = {
                    'texts': train_texts,
                    'labels': train_labels,
                    'val_texts': val_texts,
                    'val_labels': val_labels,
                    'task_type': self.task_types[task_name],
                    'distribution': {'data': len(train_labels), 'validation': len(val_labels)}
                }

        # Summary of all data
        total_train_samples = sum(len(task_data['texts']) for task_data in self.task_data.values())
        total_val_samples = sum(len(task_data.get('val_texts', [])) for task_data in self.task_data.values())
        logger.info(f"Dataset Summary: Total Training={total_train_samples}, Total Validation={total_val_samples}, Total={total_train_samples + total_val_samples}")

        logger.info(f"Client {client_id} dataset initialized with {len(tasks)} tasks")

    def get_task_dataloader(self, task_name: str, batch_size: int, validation: bool = False):
        """Get DataLoader for a specific task"""
        task_data = self.task_data[task_name]

        # Use validation data if requested and available
        if validation and 'val_texts' in task_data:
            texts = task_data['val_texts']
            labels = task_data['val_labels']
        else:
            texts = task_data['texts']
            labels = task_data['labels']

        class SimpleDataset:
            def __init__(self, texts: List[str], labels: List, tokenizer, task_type: str):
                self.texts = texts
                self.labels = labels  # Fix: assign labels to self.labels
                self.tokenizer = tokenizer
                self.task_type = task_type

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
                    'labels': torch.tensor(labels[idx],
                                         dtype=torch.float if task_data['task_type'] == "regression" else torch.long)
                }

        dataset = SimpleDataset(texts, labels, self.tokenizer, task_data['task_type'])

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=not validation)  # Don't shuffle validation

    def get_task_info(self, task_name: str):
        return self.task_data[task_name]

async def run_optimized_experiment(config: OptimizedConfig, mode: str, client_id: str = None,
                                  tasks: List[str] = None, total_clients: int = 3):
    """Run optimized MTL federated experiment"""

    if mode == "server":
        server = OptimizedMTLFederatedServer(config)
        await server.start_server()

    elif mode == "client":
        if not client_id or not tasks:
            raise ValueError("Client mode requires client_id and tasks")

        client = OptimizedMultiTaskFederatedClient(client_id, tasks, config, total_clients)
        await client.run_client()

    else:
        raise ValueError("Mode must be 'server' or 'client'")

def main():
    parser = argparse.ArgumentParser(description="Optimized MTL Federated Learning System")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID")
    parser.add_argument("--tasks", nargs='+', choices=["sst2", "qqp", "stsb"],
                       help="Task names (space-separated)")
    parser.add_argument("--port", type=int, default=8771, help="Server port")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--samples", type=int, default=300, help="Samples per client")
    parser.add_argument("--total_clients", type=int, default=3, help="Total clients")

    args = parser.parse_args()

    # Create optimized configuration
    config = OptimizedConfig(
        port=args.port,
        num_rounds=args.rounds,
        data_samples_per_client=args.samples,
        max_clients=args.total_clients,
        data_distribution="non_iid",
        non_iid_alpha=0.5
    )

    logger.info(f"Starting optimized MTL system in {args.mode} mode")

    # Run experiment
    asyncio.run(run_optimized_experiment(
        config, args.mode, args.client_id, args.tasks, args.total_clients
    ))

if __name__ == "__main__":
    main()
