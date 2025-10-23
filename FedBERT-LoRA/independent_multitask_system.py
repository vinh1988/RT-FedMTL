#!/usr/bin/env python3
"""
Independent Multi-Task Learning System WITHOUT Federated Learning
Focus: Independent training with comprehensive metrics collection via WebSocket
Models: BERT-base teacher model, tiny-BERT client models
Tasks: Multi-task learning across SST2, QQP, STSB with knowledge distillation
Output: CSV metrics for research analysis with comprehensive ML metrics
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
        logging.FileHandler('independent_multitask.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IndependentConfig:
    """Configuration for independent multi-task learning"""
    def __init__(self, **kwargs):
        # Model settings
        self.teacher_model = kwargs.get("teacher_model", "bert-base-uncased")
        self.student_model = kwargs.get("student_model", "prajjwal1/bert-tiny")

        # Training settings
        self.num_rounds = int(kwargs.get("num_rounds", 22))
        self.min_clients = int(kwargs.get("min_clients", 1))  # Changed from 2 to 1 for testing
        self.max_clients = int(kwargs.get("max_clients", 10))
        self.local_epochs = int(kwargs.get("local_epochs", 3))
        self.batch_size = int(kwargs.get("batch_size", 16))
        self.learning_rate = float(kwargs.get("learning_rate", 5e-5))  # Increased for tiny models
        self.weight_decay = float(kwargs.get("weight_decay", 0.01))

        # Knowledge Distillation settings
        self.distillation_temperature = float(kwargs.get("distillation_temperature", 4.0))
        self.distillation_alpha = float(kwargs.get("distillation_alpha", 0.7))

        # Data settings
        self.samples_per_client = int(kwargs.get("samples_per_client", 2000))
        self.max_samples_per_client = int(kwargs.get("max_samples_per_client", 2000))
        self.data_samples_per_client = min(self.samples_per_client, self.max_samples_per_client)
        self.data_distribution = kwargs.get("data_distribution", "non_iid")
        self.non_iid_alpha = float(kwargs.get("non_iid_alpha", 0.5))
        self.balance_classes = bool(kwargs.get("balance_classes", True))
        self.oversample_minority = bool(kwargs.get("oversample_minority", True))
        self.normalize_weights = bool(kwargs.get("normalize_weights", True))

        # Multi-task settings
        self.tasks = kwargs.get("tasks", ["sst2", "qqp", "stsb"])
        self.task_weights = kwargs.get("task_weights", {})  # For weighted multi-task learning

        # Communication settings
        self.port = int(kwargs.get("port", 8772))  # Different port from FL system
        self.timeout = int(kwargs.get("timeout", 300))

@dataclass
class IndependentTrainingMetrics:
    """Enhanced metrics for independent multi-task training"""
    client_id: str
    round_num: int
    task_name: str

    # Training data
    participated: bool
    data_samples: int
    data_distribution: Dict[str, int]
    data_heterogeneity_score: float

    # Performance metrics per task
    task_accuracies: Dict[str, float]
    task_precisions: Dict[str, float]
    task_recalls: Dict[str, float]
    task_f1_scores: Dict[str, float]
    task_losses: Dict[str, float]

    # Knowledge distillation metrics
    kd_loss: float
    task_loss: float
    total_loss: float

    # Per-task detailed metrics
    per_task_per_class_metrics: Dict[str, Dict[str, float]]

    # Communication metrics
    upload_time: float
    download_time: float
    parameter_size_bytes: int

    # Resource usage
    training_time: float
    memory_usage_mb: float
    cpu_utilization: float

@dataclass
class MultiTaskMetrics:
    """Multi-task learning specific metrics"""
    round_num: int

    # Overall performance across tasks
    average_task_accuracy: float
    task_accuracy_variance: float
    best_task_performance: str
    worst_task_performance: str

    # Task balance and interference
    task_interference_score: float  # How much tasks interfere with each other
    task_balance_score: float       # How balanced performance is across tasks

    # Knowledge distillation effectiveness
    average_kd_effectiveness: float
    kd_improvement_over_baseline: float

    # Client participation across tasks
    clients_per_task: Dict[str, int]
    task_completion_rates: Dict[str, float]

@dataclass
class SystemScalabilityMetrics:
    """System scalability metrics for independent training"""
    num_clients: int
    round_num: int

    # Performance scaling
    average_training_time: float
    training_time_std: float

    # Resource scaling
    total_memory_usage_mb: float
    average_memory_per_client_mb: float
    peak_cpu_utilization: float

    # Communication scaling
    total_communication_time: float
    average_upload_time: float
    total_data_processed: int

    # Multi-task efficiency
    average_tasks_per_client: float
    task_switching_overhead: float

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning with configurable distributions"""

    def __init__(self, tasks: List[str], tokenizer, client_id: int,
                 total_clients: int, samples_per_client: int,
                 distribution_type: str = "non_iid", alpha: float = 0.5):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.client_id = client_id
        self.total_clients = total_clients
        self.distribution_type = distribution_type
        self.samples_per_client = samples_per_client

        # Load datasets for all tasks
        self.task_datasets = {}
        self.task_types = {}
        self.task_num_classes = {}

        for task_name in tasks:
            if task_name == "sst2":
                dataset = load_dataset("glue", "sst2")["train"]
                texts = [item["sentence"] for item in dataset]
                labels = [item["label"] for item in dataset]
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "classification"
                self.task_num_classes[task_name] = 2
            elif task_name == "qqp":
                dataset = load_dataset("glue", "qqp")["train"]
                texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
                labels = [item["label"] for item in dataset]
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "classification"
                self.task_num_classes[task_name] = 2
            elif task_name == "stsb":
                dataset = load_dataset("glue", "stsb")["train"]
                texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
                labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to [0,1]
                self.task_datasets[task_name] = (texts, labels)
                self.task_types[task_name] = "regression"
                self.task_num_classes[task_name] = 1

        # Create task-specific data splits
        self.task_data = {}
        samples_per_task = samples_per_client // len(tasks)

        for task_name in tasks:
            texts, labels = self.task_datasets[task_name]

            if distribution_type == "non_iid":
                task_texts, task_labels, distribution = self._create_task_non_iid_split(
                    texts, labels, task_name, client_id, total_clients, samples_per_task, alpha
                )
            else:  # IID
                task_texts, task_labels, distribution = self._create_task_iid_split(
                    texts, labels, samples_per_task
                )

            self.task_data[task_name] = {
                'texts': task_texts,
                'labels': task_labels,
                'distribution': distribution,
                'task_type': self.task_types[task_name],
                'num_classes': self.task_num_classes[task_name]
            }

        logger.info(f"Client {client_id} initialized with {len(tasks)} tasks, {samples_per_client} samples per client")

    def _create_task_non_iid_split(self, texts: List[str], labels: List,
                                 task_name: str, client_id: int, total_clients: int,
                                 samples_per_task: int, alpha: float) -> Tuple[List[str], List, Dict]:
        """Create Non-IID split for a specific task"""
        np.random.seed(42 + client_id + hash(task_name) % 1000)

        task_type = "classification" if task_name != "stsb" else "regression"

        if task_type == "regression":
            return self._create_regression_non_iid(texts, labels, client_id, total_clients, samples_per_task)

        # Classification: Use Dirichlet distribution
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        proportions = np.random.dirichlet([alpha] * len(label_to_indices))

        client_indices = []
        label_counts = {}

        for label_idx, proportion in enumerate(proportions):
            if label_idx in label_to_indices:
                n_samples = int(proportion * samples_per_task)
                available_indices = label_to_indices[label_idx]

                if len(available_indices) >= n_samples:
                    selected_indices = np.random.choice(available_indices, n_samples, replace=False)
                else:
                    selected_indices = available_indices

                client_indices.extend(selected_indices)
                label_counts[f"class_{label_idx}"] = len(selected_indices)

        # Shuffle and limit
        np.random.shuffle(client_indices)
        client_indices = client_indices[:samples_per_task]

        client_texts = [texts[i] for i in client_indices]
        client_labels = [labels[i] for i in client_indices]

        # Recalculate actual distribution
        actual_distribution = {}
        for label in set(client_labels):
            actual_distribution[f"class_{label}"] = client_labels.count(label)

        return client_texts, client_labels, actual_distribution

    def _create_regression_non_iid(self, texts: List[str], labels: List[float],
                                 client_id: int, total_clients: int,
                                 samples_per_task: int) -> Tuple[List[str], List, Dict]:
        """Create Non-IID split for regression task"""
        # Sort by label values for balanced distribution
        sorted_pairs = sorted(zip(texts, labels), key=lambda x: x[1])

        # Create more bins for better granularity
        num_bins = 10  # Use 10 bins like the reference implementation for better distribution
        bins = np.linspace(0, 1, num_bins + 1)
        bin_edges = list(zip(bins[:-1], bins[1:]))

        # Assign to bins
        binned_data = [[] for _ in range(num_bins)]
        for text, label in sorted_pairs:
            bin_idx = min(int(label * num_bins), num_bins - 1)
            binned_data[bin_idx].append((text, label))

        # Balanced sampling across bins
        target_per_bin = samples_per_task // num_bins
        remaining_samples = samples_per_task % num_bins

        # Ensure minimum samples per bin (like the reference implementation)
        min_samples_per_bin = 5
        client_pairs = []

        # Log bin statistics
        bin_counts = [len(bin_data) for bin_data in binned_data]
        logger.info(f"Available samples per bin: {dict(zip(range(num_bins), bin_counts))}")

        # First pass: take target_per_bin from each bin (reference implementation approach)
        for bin_idx in range(num_bins):
            bin_samples = binned_data[bin_idx]
            if not bin_samples:
                continue

            # Calculate how many samples to take
            n_samples = min(target_per_bin, len(bin_samples))
            n_samples = max(n_samples, min(min_samples_per_bin, len(bin_samples)))

            # Random sample from this bin
            if n_samples > 0:
                sampled = np.random.choice(len(bin_samples), n_samples, replace=False)
                client_pairs.extend([bin_samples[i] for i in sampled])

        # Second pass: distribute remaining samples (reference implementation approach)
        while len(client_pairs) < samples_per_task:
            # Find non-empty bins that can provide more samples
            available_bins = [i for i in range(num_bins)
                            if len(binned_data[i]) > 0 and
                            len([p for p in client_pairs if p[1] >= bins[i] and p[1] < bins[i+1]]) <
                            (target_per_bin + 1)]

            if not available_bins:
                break

            # Distribute remaining samples
            for bin_idx in available_bins:
                if len(client_pairs) >= samples_per_task:
                    break
                remaining = [p for p in binned_data[bin_idx] if p not in client_pairs]
                if remaining:
                    client_pairs.append(remaining[0])

        client_texts = [pair[0] for pair in client_pairs]
        client_labels = [pair[1] for pair in client_pairs]

        # Create distribution
        hist, _ = np.histogram(client_labels, bins=bins)
        distribution = {f"bin_{i:02d}_{bin_edges[i][0]:.1f}-{bin_edges[i][1]:.1f}": int(count)
                       for i, count in enumerate(hist)}

        return client_texts, client_labels, distribution

    def _create_task_iid_split(self, texts: List[str], labels: List,
                              samples_per_task: int) -> Tuple[List[str], List, Dict]:
        """Create IID split for a specific task"""
        indices = np.random.choice(len(texts), samples_per_task, replace=False)
        client_texts = [texts[i] for i in indices]
        client_labels = [labels[i] for i in indices]

        task_type = "classification" if len(set(client_labels)) < 10 else "regression"  # Rough heuristic

        if task_type == "classification":
            distribution = {}
            for label in set(client_labels):
                distribution[f"class_{label}"] = client_labels.count(label)
        else:
            bins = np.linspace(0, 1, 6)
            hist, _ = np.histogram(client_labels, bins=bins)
            distribution = {f"bin_{i}": int(count) for i, count in enumerate(hist)}

        return client_texts, client_labels, distribution

    def get_task_dataloader(self, task_name: str, batch_size: int) -> DataLoader:
        """Get DataLoader for a specific task"""
        task_data = self.task_data[task_name]
        dataset = TaskSpecificDataset(
            task_data['texts'], task_data['labels'],
            task_data['task_type'], self.tokenizer
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_task_info(self, task_name: str) -> Dict:
        """Get information about a specific task"""
        return self.task_data[task_name]

class TaskSpecificDataset(Dataset):
    """Dataset for a specific task"""

    def __init__(self, texts: List[str], labels: List, task_type: str, tokenizer):
        self.texts = texts
        self.labels = labels
        self.task_type = task_type
        self.tokenizer = tokenizer

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
                                 dtype=torch.float if self.task_type == "regression" else torch.long)
        }

class IndependentMultiTaskClient:
    """Independent multi-task learning client with knowledge distillation"""

    def __init__(self, client_id: str, tasks: List[str], config: IndependentConfig, total_clients: int):
        self.client_id = client_id
        self.tasks = tasks
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_clients = total_clients

        # Initialize tokenizer (shared across tasks)
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)

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
                config.student_model, num_labels=num_labels
            )
            model.to(self.device)

            # Initialize optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

            self.student_models[task_name] = model
            self.student_optimizers[task_name] = optimizer

        # Initialize dataset
        client_num = int(client_id.split('_')[-1]) if '_' in client_id else 0
        self.dataset = MultiTaskDataset(
            tasks, self.tokenizer, client_num, total_clients,
            config.data_samples_per_client, config.data_distribution, config.non_iid_alpha
        )

        # Initialize teacher model (shared, used for knowledge distillation)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model, num_labels=1  # Use 1 for regression tasks like STSB
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # Teacher model is always in eval mode

        logger.info(f"Client {client_id} initialized for {len(tasks)} tasks: {tasks}")
        logger.info(f"Student model: {config.student_model}, Teacher model: {config.teacher_model}")

        # Count parameters
        total_student_params = sum(p.numel() for model in self.student_models.values()
                                 for p in model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        logger.info(f"Client {client_id}: {total_student_params:,} student params, {teacher_params:,} teacher params")

    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels, task_name):
        """Compute knowledge distillation loss for a specific task"""
        temperature = self.config.distillation_temperature
        alpha = self.config.distillation_alpha

        # Handle batch size and dimension mismatches
        min_batch_size = min(student_logits.size(0), teacher_logits.size(0), labels.size(0))
        student_logits = student_logits[:min_batch_size]
        teacher_logits = teacher_logits[:min_batch_size]
        labels = labels[:min_batch_size]

        task_type = self.dataset.task_types[task_name]

        if task_type == 'regression':
            kd_loss = F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze())
            task_loss = F.mse_loss(student_logits.squeeze(), labels.squeeze())
        else:
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
            task_loss = F.cross_entropy(student_logits, labels.long())

        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        return total_loss, kd_loss, task_loss

    async def local_training(self, teacher_logits: Optional[Dict] = None, round_num: int = 0) -> Tuple[Dict, IndependentTrainingMetrics]:
        """Perform independent multi-task training"""
        start_time = time.time()

        all_metrics = {}

        for task_name in self.tasks:
            task_start_time = time.time()
            model = self.student_models[task_name]
            optimizer = self.student_optimizers[task_name]
            dataloader = self.dataset.get_task_dataloader(task_name, self.config.batch_size)

            model.train()
            total_loss = 0.0
            total_kd_loss = 0.0
            total_task_loss = 0.0

            # Collect all predictions and labels for comprehensive metrics
            all_predictions = []
            all_labels = []

            # Training loop for this task
            for epoch in range(self.config.local_epochs):
                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = outputs.logits

                    # Get teacher logits (simulated for now, in real scenario would come from server)
                    teacher_batch_logits = self._get_teacher_logits(task_name, batch_idx)

                    # Compute loss
                    loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                        student_logits, teacher_batch_logits, labels, task_name
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()

                    # Collect predictions for metrics
                    task_type = self.dataset.task_types[task_name]
                    if task_type == "classification":
                        predictions = torch.argmax(student_logits, dim=-1)
                        all_predictions.extend(predictions.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())
                    else:
                        predictions = student_logits.squeeze()
                        all_predictions.extend(predictions.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())

            # Calculate comprehensive metrics for this task
            avg_loss = total_loss / (len(dataloader) * self.config.local_epochs)
            avg_kd_loss = total_kd_loss / (len(dataloader) * self.config.local_epochs)
            avg_task_loss = total_task_loss / (len(dataloader) * self.config.local_epochs)

            task_info = self.dataset.get_task_info(task_name)

            if task_info['task_type'] == "classification" and len(all_predictions) > 0:
                y_true = np.array(all_labels)
                y_pred = np.array(all_predictions)

                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )

                # Per-class metrics
                precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    y_true, y_pred, average=None, zero_division=0
                )

                unique_labels = np.unique(np.concatenate([y_true, y_pred]))
                per_class_metrics = {
                    'precision': {str(label): precision_per_class[i] if i < len(precision_per_class) else 0.0
                                for i, label in enumerate(unique_labels)},
                    'recall': {str(label): recall_per_class[i] if i < len(recall_per_class) else 0.0
                             for i, label in enumerate(unique_labels)},
                    'f1': {str(label): f1_per_class[i] if i < len(f1_per_class) else 0.0
                         for i, label in enumerate(unique_labels)}
                }

            elif task_info['task_type'] == "regression" and len(all_predictions) > 0:
                y_true = np.array(all_labels)
                y_pred = np.array(all_predictions)

                mse = float(mean_squared_error(y_true, y_pred))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_true, y_pred))
                r2 = float(r2_score(y_true, y_pred))

                try:
                    pearson_corr, _ = pearsonr(y_true, y_pred)
                    pearson_corr = float(pearson_corr)
                except:
                    pearson_corr = 0.0

                # For regression, use R² as accuracy equivalent (like the reference implementation)
                # since it measures how much variance is explained by the model
                accuracy = r2  # R² score as "accuracy" equivalent (matches reference)
                precision = 1.0 - mae   # Normalized MAE (lower MAE = higher precision)
                recall = pearson_corr   # Pearson correlation as "recall" equivalent (matches reference)
                f1 = rmse              # RMSE as F1 equivalent

                per_class_metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'pearson_corr': pearson_corr
                }
            else:
                accuracy = precision = recall = f1 = 0.0
                per_class_metrics = {}

            # Store metrics for this task
            all_metrics[task_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': avg_loss,
                'kd_loss': avg_kd_loss,
                'task_loss': avg_task_loss,
                'per_class_metrics': per_class_metrics,
                'data_distribution': task_info['distribution'],
                'data_samples': len(task_info['texts']),
                'training_time': time.time() - task_start_time
            }

        total_training_time = time.time() - start_time

        # Calculate system resource usage
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        # Calculate parameter size for communication metrics
        param_size_bytes = sum(p.numel() * p.element_size() for model in self.student_models.values()
                             for p in model.parameters())

        # Create comprehensive training metrics
        training_metrics = IndependentTrainingMetrics(
            client_id=self.client_id,
            round_num=round_num,
            task_name=",".join(self.tasks),  # All tasks for this client
            participated=True,
            data_samples=sum(len(self.dataset.task_data[task]['texts']) for task in self.tasks),
            data_distribution={task: self.dataset.task_data[task]['distribution'] for task in self.tasks},
            data_heterogeneity_score=0.0,  # Would need global distribution for this
            task_accuracies={task: metrics['accuracy'] for task, metrics in all_metrics.items()},
            task_precisions={task: metrics['precision'] for task, metrics in all_metrics.items()},
            task_recalls={task: metrics['recall'] for task, metrics in all_metrics.items()},
            task_f1_scores={task: metrics['f1_score'] for task, metrics in all_metrics.items()},
            task_losses={task: metrics['loss'] for task, metrics in all_metrics.items()},
            kd_loss=np.mean([metrics['kd_loss'] for metrics in all_metrics.values()]),
            task_loss=np.mean([metrics['task_loss'] for metrics in all_metrics.values()]),
            total_loss=np.mean([metrics['loss'] for metrics in all_metrics.values()]),
            per_task_per_class_metrics={task: metrics['per_class_metrics'] for task, metrics in all_metrics.items()},
            upload_time=0.0,  # Will be set by caller
            download_time=0.0,  # Will be set by caller
            parameter_size_bytes=param_size_bytes,
            training_time=total_training_time,
            memory_usage_mb=memory_usage_mb,
            cpu_utilization=cpu_percent
        )

        # Log comprehensive results
        logger.info(f"Client {self.client_id} completed independent training (Round {round_num}):")
        for task_name, metrics in all_metrics.items():
            if self.dataset.task_types[task_name] == "regression":
                mse = metrics['per_class_metrics'].get('mse', 0)
                rmse = metrics['per_class_metrics'].get('rmse', 0)
                mae = metrics['per_class_metrics'].get('mae', 0)
                r2 = metrics['per_class_metrics'].get('r2', 0)
                pearson = metrics['per_class_metrics'].get('pearson_corr', 0)
                logger.info(f"  {task_name}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Pearson={pearson:.4f}")
            else:
                logger.info(f"  {task_name}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

        return {}, training_metrics  # No parameters to share in independent mode

    def _get_teacher_logits(self, task_name: str, batch_idx: int):
        """Get teacher logits for knowledge distillation (simulated)"""
        # In a real implementation, this would come from the server
        # For now, we'll simulate teacher behavior

        task_info = self.dataset.get_task_info(task_name)
        task_type = task_info['task_type']

        # Get actual batch size from the dataset
        dataloader = self.dataset.get_task_dataloader(task_name, self.config.batch_size)
        batch_size = self.config.batch_size

        if task_type == "regression":
            # Return random regression values in [0,1] range - properly sized
            return (torch.randn(batch_size, 1) * 0.3 + 0.5).to(self.device)  # Dynamic batch size, mean 0.5, std 0.3
        else:
            # Return random classification logits - properly sized
            num_classes = task_info['num_classes']
            return torch.randn(batch_size, num_classes).to(self.device)  # Dynamic batch size and classes

    async def run_client(self, server_host: str = "localhost", server_port: int = 8772):
        """Run independent client with WebSocket connection to server for metrics collection"""
        uri = f"ws://{server_host}:{server_port}"

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                async with websockets.connect(
                    uri,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=15,
                    max_size=None,
                    compression=None
                ) as websocket:
                    # Register with server
                    registration = {
                        "type": "register",
                        "client_id": self.client_id,
                        "tasks": self.tasks,
                        "model": self.config.student_model,
                        "data_distributions": {task: self.dataset.task_data[task]['distribution'] for task in self.tasks},
                        "total_samples": sum(len(self.dataset.task_data[task]['texts']) for task in self.tasks)
                    }
                    await websocket.send(json.dumps(registration))

                    logger.info(f"Client {self.client_id} registered with server for tasks: {self.tasks}")

                    # Listen for training requests
                    async for message in websocket:
                        try:
                            data = json.loads(message)

                            if data["type"] == "train":
                                comm_start = time.time()

                                # Perform independent training (no parameter exchange)
                                _, training_metrics = await self.local_training(
                                    data.get("teacher_logits"), data["round"]
                                )

                                training_metrics.upload_time = time.time() - comm_start

                                # Send results back (metrics only)
                                response = {
                                    "type": "update",
                                    "client_id": self.client_id,
                                    "round": data["round"],
                                    "training_metrics": asdict(training_metrics)
                                }

                                await websocket.send(json.dumps(response))
                                logger.info(f"Client {self.client_id} sent metrics for round {data['round']}")

                            elif data["type"] == "finish":
                                logger.info(f"Client {self.client_id} received finish signal")
                                break

                        except json.JSONDecodeError as e:
                            logger.error(f"Client {self.client_id} JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Client {self.client_id} message processing error: {e}")
                            continue

                    logger.info(f"Client {self.client_id} completed successfully")
                    return

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.WebSocketException,
                    ConnectionRefusedError,
                    OSError) as e:
                retry_count += 1
                logger.error(f"Client {self.client_id} connection failed (attempt {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Client {self.client_id} retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Client {self.client_id} failed after {max_retries} attempts")
                    raise

            except Exception as e:
                logger.error(f"Client {self.client_id} unexpected error: {e}")
                raise

class IndependentMultiTaskServer:
    """Server for independent multi-task learning with comprehensive metrics collection"""

    def __init__(self, config: IndependentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize teacher model (for potential future use)
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model, num_labels=1  # Use 1 for regression tasks like STSB
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        # Client management
        self.connected_clients = {}
        self.client_updates = {}

        # Metrics storage
        self.training_metrics_history = []
        self.multitask_metrics_history = []
        self.scalability_metrics_history = []

        logger.info(f"Server initialized with {config.teacher_model} teacher model")
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        logger.info(f"Teacher model: {teacher_params:,} parameters")

    async def client_handler(self, websocket):
        """Handle client connections and collect metrics"""
        try:
            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "register":
                    client_id = data["client_id"]
                    self.connected_clients[client_id] = {
                        "websocket": websocket,
                        "tasks": data["tasks"],
                        "model": data["model"],
                        "data_distributions": data["data_distributions"],
                        "total_samples": data["total_samples"]
                    }
                    logger.info(f"Client {client_id} registered for tasks {data['tasks']}. Total clients: {len(self.connected_clients)}")

                elif data["type"] == "update":
                    client_id = data["client_id"]
                    round_num = data["round"]

                    # Store training metrics
                    training_metrics = IndependentTrainingMetrics(**data["training_metrics"])
                    self.training_metrics_history.append(training_metrics)

                    # Store client update for this round
                    if round_num not in self.client_updates:
                        self.client_updates[round_num] = {}

                    self.client_updates[round_num][client_id] = {
                        "training_metrics": training_metrics
                    }

                    logger.info(f"Received metrics from client {client_id} for round {round_num}")

        except websockets.exceptions.ConnectionClosed:
            for client_id, client_info in list(self.connected_clients.items()):
                if client_info["websocket"] == websocket:
                    del self.connected_clients[client_id]
                    logger.info(f"Client {client_id} disconnected")
                    break
        except Exception as e:
            logger.error(f"Client handler error: {e}")

    async def run_independent_training(self):
        """Run independent multi-task training with metrics collection"""

        # Wait for minimum number of clients
        while len(self.connected_clients) < self.config.min_clients:
            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{self.config.min_clients})")
            await asyncio.sleep(2)

        logger.info(f"Starting independent training with {len(self.connected_clients)} clients")

        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            round_start_time = time.time()

            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")

            # Send lightweight training request (no parameters, just round info)
            training_request = {
                "type": "train",
                "round": round_num,
                "teacher_logits": {}  # No teacher logits for independent training
            }

            logger.info(f"Sending training request for round {round_num}")

            # Send to all clients
            connected_clients_copy = dict(self.connected_clients)
            active_clients = []

            for client_id, client_info in connected_clients_copy.items():
                try:
                    websocket = client_info["websocket"]
                    if hasattr(websocket, 'closed') and websocket.closed:
                        logger.warning(f"Client {client_id} websocket is closed, removing")
                        if client_id in self.connected_clients:
                            del self.connected_clients[client_id]
                        continue

                    await asyncio.wait_for(
                        client_info["websocket"].send(json.dumps(training_request)),
                        timeout=10.0
                    )
                    active_clients.append(client_id)
                    logger.info(f"Sent training request to {client_id}")

                except asyncio.TimeoutError:
                    logger.error(f"Timeout sending training request to {client_id}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                except Exception as e:
                    logger.error(f"Failed to send training request to {client_id}: {e}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]

            logger.info(f"Successfully sent requests to {len(active_clients)} clients: {active_clients}")

            if len(active_clients) < self.config.min_clients:
                logger.warning(f"Not enough active clients ({len(active_clients)} < {self.config.min_clients})")
                continue

            # Wait for client updates
            logger.info("Waiting for client training completion...")
            timeout_count = 0
            expected_clients = len(active_clients)

            while (round_num not in self.client_updates or
                   len(self.client_updates[round_num]) < expected_clients):
                await asyncio.sleep(1)
                timeout_count += 1

                if timeout_count % 10 == 0:
                    received = len(self.client_updates.get(round_num, {}))
                    logger.info(f"Round {round_num}: Received {received}/{expected_clients} client updates")

                if timeout_count > self.config.timeout:
                    logger.warning(f"Timeout waiting for clients in round {round_num}")
                    break

            # Process round results and calculate metrics
            if round_num in self.client_updates:
                # Calculate multi-task metrics
                multitask_metrics = self.calculate_multitask_metrics(round_num, self.client_updates[round_num])
                scalability_metrics = self.calculate_scalability_metrics(
                    round_num, self.client_updates[round_num], round_start_time
                )

                # Store metrics
                self.multitask_metrics_history.append(multitask_metrics)
                self.scalability_metrics_history.append(scalability_metrics)

                logger.info(f"Round {round_num} completed in {time.time() - round_start_time:.2f}s")
                logger.info(f"Clients: {len(self.client_updates[round_num])}, "
                           f"Avg Task Accuracy: {multitask_metrics.average_task_accuracy:.4f}")

        # Finish training
        finish_message = {"type": "finish"}
        connected_clients_copy = dict(self.connected_clients)
        for client_id, client_info in connected_clients_copy.items():
            try:
                await client_info["websocket"].send(json.dumps(finish_message))
            except Exception as e:
                logger.error(f"Failed to send finish message to {client_id}: {e}")

        # Save all metrics to CSV
        self.save_metrics_to_csv()

        logger.info("Independent training completed")

    def calculate_multitask_metrics(self, round_num: int, client_updates: Dict[str, Dict]) -> MultiTaskMetrics:
        """Calculate multi-task learning specific metrics"""

        # Collect task performances across all clients
        task_performances = defaultdict(list)
        task_completion_rates = defaultdict(int)
        total_clients = len(client_updates)

        for client_id, update in client_updates.items():
            metrics = update["training_metrics"]

            # Count completion per task
            for task_name in metrics.task_accuracies.keys():
                task_performances[task_name].append(metrics.task_accuracies[task_name])
                task_completion_rates[task_name] += 1

        # Calculate average performance per task
        avg_task_performances = {
            task: np.mean(perfs) if perfs else 0.0
            for task, perfs in task_performances.items()
        }

        # Overall metrics
        all_performances = [perf for perfs in task_performances.values() for perf in perfs]
        average_task_accuracy = np.mean(all_performances) if all_performances else 0.0
        task_accuracy_variance = np.var(all_performances) if len(all_performances) > 1 else 0.0

        # Best and worst performing tasks
        if avg_task_performances:
            best_task = max(avg_task_performances.items(), key=lambda x: x[1])
            worst_task = min(avg_task_performances.items(), key=lambda x: x[1])
            best_task_performance = best_task[0]
            worst_task_performance = worst_task[0]
        else:
            best_task_performance = worst_task_performance = "none"

        # Task completion rates
        task_completion_rates = {
            task: count / total_clients for task, count in task_completion_rates.items()
        }

        # Calculate task interference (variance in performance across tasks)
        task_performances_per_client = defaultdict(dict)
        for client_id, update in client_updates.items():
            metrics = update["training_metrics"]
            for task_name, accuracy in metrics.task_accuracies.items():
                task_performances_per_client[client_id][task_name] = accuracy

        # Calculate interference as average variance across clients
        interference_scores = []
        for client_perfs in task_performances_per_client.values():
            if len(client_perfs) > 1:
                interference_scores.append(np.var(list(client_perfs.values())))

        task_interference_score = np.mean(interference_scores) if interference_scores else 0.0

        # Task balance (how equal performance is across tasks)
        if avg_task_performances:
            task_balance_score = 1 - np.std(list(avg_task_performances.values())) / (np.mean(list(avg_task_performances.values())) + 1e-10)
        else:
            task_balance_score = 0.0

        return MultiTaskMetrics(
            round_num=round_num,
            average_task_accuracy=average_task_accuracy,
            task_accuracy_variance=task_accuracy_variance,
            best_task_performance=best_task_performance,
            worst_task_performance=worst_task_performance,
            task_interference_score=task_interference_score,
            task_balance_score=task_balance_score,
            average_kd_effectiveness=0.0,  # Would need baseline comparison
            kd_improvement_over_baseline=0.0,
            clients_per_task={task: count for task, count in task_completion_rates.items()},
            task_completion_rates=task_completion_rates
        )

    def calculate_scalability_metrics(self, round_num: int, client_updates: Dict[str, Dict],
                                   start_time: float) -> SystemScalabilityMetrics:
        """Calculate system scalability metrics"""

        num_clients = len(client_updates)
        if num_clients == 0:
            return SystemScalabilityMetrics(
                num_clients=0, round_num=round_num,
                average_training_time=0.0, training_time_std=0.0,
                total_memory_usage_mb=0.0, average_memory_per_client_mb=0.0,
                peak_cpu_utilization=0.0, total_communication_time=0.0,
                average_upload_time=0.0, total_data_processed=0,
                average_tasks_per_client=0.0, task_switching_overhead=0.0
            )

        # Collect metrics
        training_times = []
        memory_usages = []
        upload_times = []
        data_samples = []
        tasks_per_client = []

        for update in client_updates.values():
            metrics = update["training_metrics"]
            training_times.append(metrics.training_time)
            memory_usages.append(metrics.memory_usage_mb)
            upload_times.append(metrics.upload_time)
            data_samples.append(metrics.data_samples)
            tasks_per_client.append(len(metrics.task_accuracies))

        # Calculate resource usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        total_memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        return SystemScalabilityMetrics(
            num_clients=num_clients,
            round_num=round_num,
            average_training_time=np.mean(training_times),
            training_time_std=np.std(training_times),
            total_memory_usage_mb=total_memory_mb,
            average_memory_per_client_mb=np.mean(memory_usages),
            peak_cpu_utilization=cpu_percent,
            total_communication_time=time.time() - start_time,
            average_upload_time=np.mean(upload_times),
            total_data_processed=sum(data_samples),
            average_tasks_per_client=np.mean(tasks_per_client),
            task_switching_overhead=0.0  # Would need more detailed timing
        )

    def save_metrics_to_csv(self):
        """Save all metrics to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        experiment_id = f"independent_{len(self.config.tasks)}tasks_{self.config.max_clients}c_{timestamp}"
        results_dir = Path("independent_multitask_results")
        results_dir.mkdir(exist_ok=True)

        # Save training metrics
        training_file = results_dir / f"training_metrics_{experiment_id}.csv"
        with open(training_file, 'w', newline='') as f:
            if self.training_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.training_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.training_metrics_history:
                    row = asdict(metrics)
                    # Convert dict fields to JSON strings for CSV
                    for field in ['task_accuracies', 'task_precisions', 'task_recalls',
                                'task_f1_scores', 'task_losses', 'per_task_per_class_metrics']:
                        row[field] = json.dumps(row[field])
                    writer.writerow(row)

        # Save multi-task metrics
        multitask_file = results_dir / f"multitask_metrics_{experiment_id}.csv"
        with open(multitask_file, 'w', newline='') as f:
            if self.multitask_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.multitask_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.multitask_metrics_history:
                    row = asdict(metrics)
                    # Convert dict fields to JSON strings for CSV
                    for field in ['clients_per_task', 'task_completion_rates']:
                        row[field] = json.dumps(row[field])
                    writer.writerow(row)

        # Save scalability metrics
        scalability_file = results_dir / f"scalability_metrics_{experiment_id}.csv"
        with open(scalability_file, 'w', newline='') as f:
            if self.scalability_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.scalability_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.scalability_metrics_history:
                    writer.writerow(asdict(metrics))

        logger.info(f"Metrics saved to CSV files in {results_dir}")
        logger.info(f"Files: {training_file.name}, {multitask_file.name}, {scalability_file.name}")

    async def start_server(self):
        """Start the independent multi-task learning server"""
        logger.info(f"Starting server on port {self.config.port}")

        server = await websockets.serve(
            self.client_handler,
            "localhost",
            self.config.port,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=15,
            max_size=None,
            compression=None
        )

        logger.info(f"Server listening on localhost:{self.config.port}")

        # Run independent training
        await self.run_independent_training()

        server.close()
        await server.wait_closed()

async def run_independent_experiment(config: IndependentConfig, mode: str, client_id: str = None,
                                   tasks: List[str] = None, total_clients: int = 5):
    """Run independent multi-task learning experiment"""

    if mode == "server":
        server = IndependentMultiTaskServer(config)
        await server.start_server()

    elif mode == "client":
        if not client_id or not tasks:
            raise ValueError("Client mode requires client_id and tasks")

        client = IndependentMultiTaskClient(client_id, tasks, config, total_clients)
        await client.run_client("localhost", config.port)

    else:
        raise ValueError("Mode must be 'server' or 'client'")

def main():
    parser = argparse.ArgumentParser(description="Independent Multi-Task Learning System")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--tasks", nargs='+', default=["sst2", "qqp", "stsb"],
                       help="Task names (required for client mode)")
    parser.add_argument("--port", type=int, default=8772, help="Server port")
    parser.add_argument("--rounds", type=int, default=22, help="Number of training rounds")
    parser.add_argument("--samples", type=int, default=2000, help="Data samples per client")
    parser.add_argument("--total_clients", type=int, default=5, help="Total number of clients")
    parser.add_argument("--distribution", choices=["iid", "non_iid"], default="non_iid",
                       help="Data distribution type")
    parser.add_argument("--alpha", type=float, default=0.5, help="Non-IID alpha parameter")

    args = parser.parse_args()

    logger.info(f"Using configuration: {args}")

    # Create configuration
    config = IndependentConfig(
        port=args.port,
        num_rounds=args.rounds,
        data_samples_per_client=args.samples,
        data_distribution=args.distribution,
        non_iid_alpha=args.alpha,
        max_clients=args.total_clients,
        tasks=args.tasks
    )

    # Run experiment
    asyncio.run(run_independent_experiment(
        config, args.mode, args.client_id, args.tasks, args.total_clients
    ))

if __name__ == "__main__":
    main()
