#!/usr/bin/env python3
"""
Federated Streaming System WITHOUT LoRA - Enhanced with F1/Precision Metrics
Focus: Non-IID data metrics and client participation analysis (2-10 clients)
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
        logging.FileHandler('no_lora_federated.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NoLoRAConfig:
    """Configuration for non-LoRA federated learning"""
    def __init__(self, **kwargs):
        # Model settings
        self.server_model = kwargs.get("server_model", "bert-base-uncased")
        self.client_model = kwargs.get("client_model", "prajjwal1/bert-tiny")
        
        # Training settings
        self.num_rounds = int(kwargs.get("num_rounds", 22))
        self.min_clients = int(kwargs.get("min_clients", 2))
        self.max_clients = int(kwargs.get("max_clients", 10))
        self.local_epochs = int(kwargs.get("local_epochs", 3))
        self.batch_size = int(kwargs.get("batch_size", 16))
        self.learning_rate = float(kwargs.get("learning_rate", 2e-5))
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
        
        # Communication settings
        self.port = int(kwargs.get("port", 8771))
        self.timeout = int(kwargs.get("timeout", 300))

@dataclass
class ClientParticipationMetrics:
    """Enhanced client participation metrics with comprehensive ML metrics"""
    client_id: str
    round_num: int
    
    # Participation data
    participated: bool
    participation_rate: float  # Historical participation rate
    consecutive_participations: int
    total_participations: int
    
    # Data characteristics
    data_samples: int
    data_distribution: Dict[str, int]  # Label distribution
    data_heterogeneity_score: float  # Measure of how different from global
    
    # Enhanced performance metrics
    local_accuracy: float
    local_precision: float  # Weighted average precision
    local_recall: float     # Weighted average recall
    local_f1_score: float   # Weighted average F1-score
    local_loss: float
    contribution_weight: float  # Weight in aggregation
    
    # Per-class metrics (for detailed analysis)
    per_class_precision: Dict[str, float]  # Precision per class
    per_class_recall: Dict[str, float]     # Recall per class
    per_class_f1: Dict[str, float]         # F1-score per class
    confusion_matrix_flat: List[int]       # Flattened confusion matrix
    
    # Communication metrics
    upload_time: float
    download_time: float
    parameter_size_bytes: int

@dataclass
class NonIIDMetrics:
    """Non-IID specific metrics"""
    round_num: int
    
    # Data distribution analysis
    global_label_distribution: Dict[str, float]
    client_label_distributions: Dict[str, Dict[str, float]]
    
    # Heterogeneity measures
    kl_divergence_scores: Dict[str, float]  # Per client KL from global
    jensen_shannon_divergence: float  # Overall heterogeneity
    earth_movers_distance: float  # Distribution distance
    
    # Performance under heterogeneity
    accuracy_variance: float  # Variance across clients
    convergence_rate: float
    fairness_score: float  # Performance equality across clients
    
    # Aggregation quality
    parameter_diversity: float
    consensus_measure: float
    aggregation_efficiency: float

@dataclass
class ScalabilityMetrics:
    """Client scalability metrics"""
    num_clients: int
    round_num: int
    
    # Performance scaling
    average_accuracy: float
    accuracy_std: float
    worst_client_accuracy: float
    best_client_accuracy: float
    
    # Enhanced ML metrics
    average_precision: float
    precision_std: float
    average_recall: float
    recall_std: float
    average_f1_score: float
    f1_score_std: float
    
    # Communication scaling
    total_communication_time: float
    average_client_latency: float
    aggregation_time: float
    
    # System scaling
    memory_usage_mb: float
    cpu_utilization: float
    throughput_samples_per_sec: float

class NonIIDDataset(Dataset):
    """Dataset with Non-IID distribution simulation"""
    
    def __init__(self, task_name: str, split: str, tokenizer, client_id: int, 
                 total_clients: int, samples_per_client: int, 
                 distribution_type: str = "non_iid", alpha: float = 0.5):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.client_id = client_id
        self.distribution_type = distribution_type
        
        # Load full dataset
        if task_name == "sst2":
            dataset = load_dataset("glue", "sst2")[split]
            texts = [item["sentence"] for item in dataset]
            labels = [item["label"] for item in dataset]
            self.task_type = "classification"
            self.num_classes = 2
        elif task_name == "qqp":
            dataset = load_dataset("glue", "qqp")[split]
            texts = [f"{item['question1']} [SEP] {item['question2']}" for item in dataset]
            labels = [item["label"] for item in dataset]
            self.task_type = "classification"
            self.num_classes = 2
        elif task_name == "stsb":
            dataset = load_dataset("glue", "stsb")[split]
            texts = [f"{item['sentence1']} [SEP] {item['sentence2']}" for item in dataset]
            labels = [float(item["label"]) / 5.0 for item in dataset]  # Normalize to [0,1]
            self.task_type = "regression"
            self.num_classes = 1
        
        # Create Non-IID distribution
        if distribution_type == "non_iid":
            self.texts, self.labels, self.label_distribution = self._create_non_iid_split(
                texts, labels, client_id, total_clients, samples_per_client, alpha
            )
        elif distribution_type == "pathological":
            self.texts, self.labels, self.label_distribution = self._create_pathological_split(
                texts, labels, client_id, total_clients, samples_per_client
            )
        else:  # IID
            self.texts, self.labels, self.label_distribution = self._create_iid_split(
                texts, labels, samples_per_client
            )
        
        logger.info(f"Client {client_id} ({task_name}): {len(self.texts)} samples, "
                   f"distribution: {self.label_distribution}")
    
    def _create_non_iid_split(self, texts: List[str], labels: List, 
                            client_id: int, total_clients: int, 
                            samples_per_client: int, alpha: float) -> Tuple[List[str], List, Dict]:
        """Create Non-IID split using Dirichlet distribution"""
        
        if self.task_type == "regression":
            # For regression, create heterogeneity by value ranges
            return self._create_regression_non_iid(texts, labels, client_id, total_clients, samples_per_client)
        
        # Classification: Use Dirichlet distribution
        np.random.seed(42 + client_id)  # Reproducible but different per client
        
        # Group data by labels
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        
        # Generate Dirichlet distribution for this client
        proportions = np.random.dirichlet([alpha] * self.num_classes)
        
        client_indices = []
        label_counts = {}
        
        for label_idx, proportion in enumerate(proportions):
            if label_idx in label_to_indices:
                n_samples = int(proportion * samples_per_client)
                available_indices = label_to_indices[label_idx]
                
                if len(available_indices) >= n_samples:
                    selected_indices = np.random.choice(available_indices, n_samples, replace=False)
                else:
                    selected_indices = available_indices  # Take all available
                
                client_indices.extend(selected_indices)
                label_counts[f"class_{label_idx}"] = len(selected_indices)
        
        # Shuffle and limit to requested size
        np.random.shuffle(client_indices)
        client_indices = client_indices[:samples_per_client]
        
        client_texts = [texts[i] for i in client_indices]
        client_labels = [labels[i] for i in client_indices]
        
        # Recalculate actual distribution
        actual_distribution = {}
        for label in set(client_labels):
            actual_distribution[f"class_{label}"] = client_labels.count(label)
        
        return client_texts, client_labels, actual_distribution
    
    def _create_regression_non_iid(self, texts: List[str], labels: List[float], 
                                 client_id: int, total_clients: int, 
                                 samples_per_client: int) -> Tuple[List[str], List, Dict]:
        """
        Create Non-IID split for regression with balanced bin distribution.
        Ensures good representation across the entire label range.
        """
        # Sort by label values
        sorted_pairs = sorted(zip(texts, labels), key=lambda x: x[1])
        
        # Create more bins for better granularity
        num_bins = 10
        bins = np.linspace(0, 1, num_bins + 1)
        bin_edges = list(zip(bins[:-1], bins[1:]))
        
        # Assign each sample to a bin
        binned_data = [[] for _ in range(num_bins)]
        for text, label in sorted_pairs:
            bin_idx = min(int(label * num_bins), num_bins - 1)
            binned_data[bin_idx].append((text, label))
        
        # Calculate target samples per bin for balanced distribution
        target_per_bin = samples_per_client // num_bins
        remaining_samples = samples_per_client % num_bins
        
        # Ensure minimum samples per bin
        min_samples_per_bin = 5
        client_pairs = []
        
        # Log bin statistics
        bin_counts = [len(bin_data) for bin_data in binned_data]
        logger.info(f"Available samples per bin: {dict(zip(range(num_bins), bin_counts))}")
        
        # First pass: take target_per_bin from each bin
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
        
        # Second pass: distribute remaining samples
        while len(client_pairs) < samples_per_client:
            # Find non-empty bins that can provide more samples
            available_bins = [i for i in range(num_bins) 
                            if len(binned_data[i]) > 0 and 
                            len([p for p in client_pairs if p[1] >= bins[i] and p[1] < bins[i+1]]) < 
                            (target_per_bin + 1)]
            
            if not available_bins:
                break
                
            # Distribute remaining samples
            for bin_idx in available_bins:
                if len(client_pairs) >= samples_per_client:
                    break
                remaining = [p for p in binned_data[bin_idx] if p not in client_pairs]
                if remaining:
                    client_pairs.append(remaining[0])
        
        # Shuffle and limit to requested size
        np.random.shuffle(client_pairs)
        client_pairs = client_pairs[:samples_per_client]
        
        client_texts = [pair[0] for pair in client_pairs]
        client_labels = [pair[1] for pair in client_pairs]
        
        # Create distribution bins for logging
        hist, _ = np.histogram(client_labels, bins=bins)
        distribution = {f"bin_{i:02d}_{bin_edges[i][0]:.1f}-{bin_edges[i][1]:.1f}": int(count) 
                       for i, count in enumerate(hist)}
        
        # Log final distribution
        logger.info(f"Final distribution for client {client_id}: {distribution}")
        
        return client_texts, client_labels, distribution
    
    def _create_iid_split(self, texts: List[str], labels: List, 
                        samples_per_client: int) -> Tuple[List[str], List, Dict]:
        """Create IID split (random sampling)"""
        
        indices = np.random.choice(len(texts), samples_per_client, replace=False)
        client_texts = [texts[i] for i in indices]
        client_labels = [labels[i] for i in indices]
        
        if self.task_type == "classification":
            distribution = {}
            for label in set(client_labels):
                distribution[f"class_{label}"] = client_labels.count(label)
        else:
            bins = np.linspace(0, 1, 6)
            hist, _ = np.histogram(client_labels, bins=bins)
            distribution = {f"bin_{i}": int(count) for i, count in enumerate(hist)}
        
        return client_texts, client_labels, distribution
    
    def _create_pathological_split(self, texts: List[str], labels: List, 
                                 client_id: str, total_clients: int, 
                                 samples_per_client: int) -> Tuple[List[str], List, Dict]:
        """Create pathological split (each client gets only one class)"""
        
        # Get unique labels/classes
        unique_labels = list(set(labels))
        
        # Assign each client to a specific class (round-robin)
        client_num = int(client_id.split('_')[-1]) - 1  # Extract client number
        assigned_label = unique_labels[client_num % len(unique_labels)]
        
        # Find all samples with the assigned label
        label_indices = [i for i, label in enumerate(labels) if label == assigned_label]
        
        # Sample from this label only
        if len(label_indices) >= samples_per_client:
            selected_indices = np.random.choice(label_indices, samples_per_client, replace=False)
        else:
            # If not enough samples, take all and pad with repetition
            selected_indices = np.random.choice(label_indices, samples_per_client, replace=True)
        
        client_texts = [texts[i] for i in selected_indices]
        client_labels = [labels[i] for i in selected_indices]
        
        # Create distribution
        if self.task_type == "classification":
            distribution = {}
            for label in set(client_labels):
                distribution[f"class_{label}"] = client_labels.count(label)
        else:
            bins = np.linspace(0, 1, 6)
            hist, _ = np.histogram(client_labels, bins=bins)
            distribution = {f"bin_{i}": int(count) for i, count in enumerate(hist)}
        
        return client_texts, client_labels, distribution
    
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
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get the label distribution for this client"""
        return self.label_distribution
    
    def calculate_heterogeneity_score(self, global_distribution: Dict[str, float]) -> float:
        """Calculate KL divergence from global distribution"""
        
        # Normalize local distribution
        total_samples = sum(self.label_distribution.values())
        local_dist = {k: v / total_samples for k, v in self.label_distribution.items()}
        
        # Calculate KL divergence
        kl_div = 0.0
        for key in global_distribution:
            if key in local_dist and local_dist[key] > 0 and global_distribution[key] > 0:
                kl_div += local_dist[key] * np.log(local_dist[key] / global_distribution[key])
        
        return kl_div

class NoLoRAFederatedClient:
    """Federated client without LoRA - full parameter training"""
    
    def __init__(self, client_id: str, task_name: str, config: NoLoRAConfig, 
                 total_clients: int):
        self.client_id = client_id
        self.task_name = task_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_clients = total_clients
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)
        
        # Determine task type
        self.task_type = "regression" if task_name == "stsb" else "classification"
        
        # Initialize model
        if self.task_type == "regression":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=1
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=2
            )
        
        self.model.to(self.device)
        
        # Extract client number from client_id for Non-IID distribution
        client_num = int(client_id.split('_')[-1]) if '_' in client_id else 0
        
        # Initialize Non-IID dataset
        self.dataset = NonIIDDataset(
            task_name, "train", self.tokenizer, client_num, 
            total_clients, config.data_samples_per_client,
            config.data_distribution, config.non_iid_alpha
        )
        
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        # Initialize optimizer (all parameters)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Participation tracking
        self.participation_history = []
        self.total_participations = 0
        
        logger.info(f"Client {client_id} initialized for task {task_name}")
        logger.info(f"Model: {config.client_model}, Full parameter training")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Client {client_id}: {total_params:,} total params, {trainable_params:,} trainable")
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss"""
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
            kd_loss = F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze())
            task_loss = F.mse_loss(student_logits.squeeze(), labels)
        else:
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
            task_loss = F.cross_entropy(student_logits, labels.long())
        
        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        return total_loss, kd_loss, task_loss
    
    async def local_training(self, global_params: Optional[Dict] = None, 
                           teacher_logits: Optional[Dict] = None, 
                           round_num: int = 0) -> Tuple[Dict, ClientParticipationMetrics]:
        """Perform local training with full parameters"""
        start_time = time.time()
        
        # Set global parameters if provided (for Non-LoRA, we simulate independent training)
        if global_params and len(global_params) > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        param.data = torch.tensor(global_params[name]).to(self.device)
        else:
            logger.info(f"Client {self.client_id} training independently (no global params received)")
        
        self.model.train()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        
        # Collect all predictions and labels for comprehensive metrics
        all_predictions = []
        all_labels = []
        
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
                    teacher_batch_logits = torch.tensor(teacher_logits[str(batch_idx)]).to(self.device)
                    loss, kd_loss, task_loss = self.knowledge_distillation_loss(
                        student_logits, teacher_batch_logits, labels
                    )
                    total_kd_loss += kd_loss.item()
                    total_task_loss += task_loss.item()
                else:
                    if self.task_type == "regression":
                        loss = F.mse_loss(student_logits.squeeze(), labels)
                    else:
                        loss = F.cross_entropy(student_logits, labels.long())
                    total_task_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Collect predictions and labels for comprehensive metrics
                if self.task_type == "classification":
                    predictions = torch.argmax(student_logits, dim=-1)
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
                else:
                    # For regression, we'll convert to binary for metrics
                    predictions = student_logits.squeeze()
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
        
        training_time = time.time() - start_time
        
        # Get updated parameters (subset for metrics collection to avoid huge messages)
        updated_params = {}
        param_count = 0
        for name, param in self.model.named_parameters():
            if param_count < 3:  # Only send first 3 parameters for metrics
                updated_params[name] = param.data.cpu().numpy()
                param_count += 1
        
        logger.info(f"Client {self.client_id} sending {len(updated_params)} parameter groups for metrics")
        
        # Calculate comprehensive metrics using sklearn
        avg_loss = total_loss / (len(self.dataloader) * self.config.local_epochs)
        
        if self.task_type == "classification" and len(all_predictions) > 0:
            # Convert to numpy arrays
            y_true = np.array(all_labels)
            y_pred = np.array(all_predictions)
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_flat = cm.flatten().tolist()
            
            # Create per-class dictionaries
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            per_class_precision = {str(label): precision_per_class[i] if i < len(precision_per_class) else 0.0 
                                 for i, label in enumerate(unique_labels)}
            per_class_recall = {str(label): recall_per_class[i] if i < len(recall_per_class) else 0.0 
                              for i, label in enumerate(unique_labels)}
            per_class_f1 = {str(label): f1_per_class[i] if i < len(f1_per_class) else 0.0 
                          for i, label in enumerate(unique_labels)}
            
        elif self.task_type == "regression" and len(all_predictions) > 0:
            # Regression-specific metrics
            y_true = np.array(all_labels)
            y_pred = np.array(all_predictions)
            
            # Calculate regression metrics
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            
            # Pearson correlation
            try:
                pearson_corr, _ = pearsonr(y_true, y_pred)
                pearson_corr = float(pearson_corr)
            except:
                pearson_corr = 0.0
            
            # For regression, store metrics in classification fields for compatibility
            # but use meaningful regression values
            accuracy = r2  # R² score as "accuracy" equivalent
            precision = 1.0 - mae  # Inverse of MAE (higher is better)
            recall = pearson_corr  # Correlation as "recall" equivalent
            f1 = rmse  # RMSE stored in F1 field
            
            per_class_precision = {"mse": mse, "rmse": rmse, "mae": mae}
            per_class_recall = {"r2": r2, "pearson": pearson_corr}
            per_class_f1 = {}
            cm_flat = []
            
        else:
            # Fallback for empty predictions
            accuracy = 0.0
            precision = recall = f1 = 0.0
            per_class_precision = per_class_recall = per_class_f1 = {}
            cm_flat = []
        
        # Update participation tracking
        self.total_participations += 1
        self.participation_history.append(round_num)
        participation_rate = self.total_participations / max(1, round_num)
        
        # Calculate parameter size for communication metrics
        param_size_bytes = sum(param.nbytes for param in updated_params.values())
        
        # Calculate heterogeneity score (will be updated with global distribution)
        heterogeneity_score = 0.0  # Placeholder
        
        # Create enhanced participation metrics
        participation_metrics = ClientParticipationMetrics(
            client_id=self.client_id,
            round_num=round_num,
            participated=True,
            participation_rate=participation_rate,
            consecutive_participations=len([r for r in self.participation_history[-5:] if r >= round_num - 4]),
            total_participations=self.total_participations,
            data_samples=len(self.dataset),
            data_distribution=self.dataset.get_label_distribution(),
            data_heterogeneity_score=heterogeneity_score,
            local_accuracy=accuracy,
            local_precision=precision,
            local_recall=recall,
            local_f1_score=f1,
            local_loss=avg_loss,
            contribution_weight=len(self.dataset),  # Weight by data size
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            confusion_matrix_flat=cm_flat,
            upload_time=0.0,  # Will be set by caller
            download_time=0.0,  # Will be set by caller
            parameter_size_bytes=param_size_bytes
        )
        
        # Log based on task type
        if self.task_type == "regression":
            # Extract regression metrics from per_class dictionaries
            mse = per_class_precision.get("mse", 0.0)
            rmse = per_class_precision.get("rmse", 0.0)
            mae = per_class_precision.get("mae", 0.0)
            r2 = per_class_recall.get("r2", 0.0)
            pearson = per_class_recall.get("pearson", 0.0)
            
            logger.info(f"Client {self.client_id} ({self.task_name}) training complete: "
                       f"Loss={avg_loss:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, "
                       f"R²={r2:.4f}, Pearson={pearson:.4f}, Params={param_size_bytes/1024/1024:.1f}MB")
        else:
            logger.info(f"Client {self.client_id} ({self.task_name}) training complete: "
                       f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
                       f"Params={param_size_bytes/1024/1024:.1f}MB")
        
        return updated_params, participation_metrics
    
    async def run_client(self, server_host: str = "localhost", server_port: int = 8771):
        """Run client with WebSocket connection to server"""
        uri = f"ws://{server_host}:{server_port}"
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Configure WebSocket with longer timeout and ping interval
                async with websockets.connect(
                    uri, 
                    ping_interval=None,  # Disable automatic ping
                    ping_timeout=None,   # Disable ping timeout
                    close_timeout=15,    # Wait 15 seconds for close
                    max_size=None,       # No message size limit
                    compression=None     # Disable compression for stability
                ) as websocket:
                    # Register with server
                    registration = {
                        "type": "register",
                        "client_id": self.client_id,
                        "task": self.task_name,
                        "model": self.config.client_model,
                        "data_distribution": self.dataset.get_label_distribution(),
                        "total_samples": len(self.dataset)
                    }
                    await websocket.send(json.dumps(registration))
                    
                    logger.info(f"Client {self.client_id} registered with server")
                    
                    # Listen for training requests
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            if data["type"] == "train":
                                comm_start = time.time()
                                
                                # Perform local training
                                updated_params, participation_metrics = await self.local_training(
                                    data.get("global_params"),
                                    data.get("teacher_logits"),
                                    data["round"]
                                )
                                
                                participation_metrics.upload_time = time.time() - comm_start
                                
                                # Send results back
                                response = {
                                    "type": "update",
                                    "client_id": self.client_id,
                                    "round": data["round"],
                                    "parameters": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                 for k, v in updated_params.items()},
                                    "participation_metrics": asdict(participation_metrics)
                                }
                                
                                await websocket.send(json.dumps(response))
                                logger.info(f"Client {self.client_id} sent update for round {data['round']}")
                            
                            elif data["type"] == "finish":
                                logger.info(f"Client {self.client_id} received finish signal")
                                break
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Client {self.client_id} JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Client {self.client_id} message processing error: {e}")
                            continue
                    
                    # If we reach here, connection was successful and completed
                    logger.info(f"Client {self.client_id} completed successfully")
                    return
                    
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.WebSocketException,
                    ConnectionRefusedError,
                    OSError) as e:
                retry_count += 1
                logger.warning(f"Client {self.client_id} connection failed (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"Client {self.client_id} retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Client {self.client_id} failed after {max_retries} attempts")
                    raise
            
            except Exception as e:
                logger.error(f"Client {self.client_id} unexpected error: {e}")
                raise

class NoLoRAFederatedServer:
    """Federated server for non-LoRA experiments with detailed metrics"""
    
    def __init__(self, config: NoLoRAConfig):
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
        self.client_updates = {}
        
        # Metrics storage
        self.participation_metrics_history = []
        self.non_iid_metrics_history = []
        self.scalability_metrics_history = []
        
        # Global data distribution (for heterogeneity calculation)
        self.global_distribution = {}
        
        logger.info(f"Server initialized with {config.server_model}")
        total_params = sum(p.numel() for p in self.global_model.parameters())
        logger.info(f"Global model: {total_params:,} parameters")
    
    async def client_handler(self, websocket):
        """Handle client connections"""
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "register":
                    client_id = data["client_id"]
                    self.connected_clients[client_id] = {
                        "websocket": websocket,
                        "task": data["task"],
                        "model": data["model"],
                        "data_distribution": data["data_distribution"],
                        "total_samples": data["total_samples"]
                    }
                    logger.info(f"Client {client_id} registered. Total clients: {len(self.connected_clients)}")
                    
                    # Update global distribution
                    self._update_global_distribution()
                
                elif data["type"] == "update":
                    client_id = data["client_id"]
                    round_num = data["round"]
                    
                    # Convert parameters back to numpy arrays
                    parameters = {k: np.array(v) for k, v in data["parameters"].items()}
                    
                    # Store participation metrics
                    participation_metrics = ClientParticipationMetrics(**data["participation_metrics"])
                    
                    # Calculate heterogeneity score
                    client_info = self.connected_clients[client_id]
                    heterogeneity_score = self._calculate_heterogeneity_score(
                        client_info["data_distribution"]
                    )
                    participation_metrics.data_heterogeneity_score = heterogeneity_score
                    
                    self.participation_metrics_history.append(participation_metrics)
                    
                    # Store client update for aggregation
                    if round_num not in self.client_updates:
                        self.client_updates[round_num] = {}
                    
                    self.client_updates[round_num][client_id] = {
                        "parameters": parameters,
                        "participation_metrics": participation_metrics
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
    
    def _update_global_distribution(self):
        """Update global data distribution from all clients"""
        total_samples = 0
        combined_distribution = defaultdict(int)
        
        for client_info in self.connected_clients.values():
            client_dist = client_info["data_distribution"]
            client_samples = client_info["total_samples"]
            
            for label, count in client_dist.items():
                combined_distribution[label] += count
                total_samples += count
        
        # Normalize to probabilities
        self.global_distribution = {
            label: count / total_samples 
            for label, count in combined_distribution.items()
        }
        
        logger.info(f"Updated global distribution: {self.global_distribution}")
    
    def _calculate_heterogeneity_score(self, client_distribution: Dict[str, int]) -> float:
        """Calculate KL divergence from global distribution"""
        if not self.global_distribution:
            return 0.0
        
        # Normalize client distribution
        total_samples = sum(client_distribution.values())
        if total_samples == 0:
            return 0.0
        
        client_dist = {k: v / total_samples for k, v in client_distribution.items()}
        
        # Calculate KL divergence
        kl_div = 0.0
        for key in self.global_distribution:
            if key in client_dist and client_dist[key] > 0 and self.global_distribution[key] > 0:
                kl_div += client_dist[key] * np.log(client_dist[key] / self.global_distribution[key])
        
        return kl_div
    
    def aggregate_parameters(self, client_updates: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Aggregate client parameters using FedAvg"""
        if not client_updates:
            return {}
        
        # Get parameter names from first client
        first_client = next(iter(client_updates.values()))
        param_names = first_client["parameters"].keys()
        
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect parameters and weights from all clients
            client_params = []
            client_weights = []
            
            for client_id, update in client_updates.items():
                if param_name in update["parameters"]:
                    client_params.append(update["parameters"][param_name])
                    # Weight by number of samples
                    client_weights.append(update["participation_metrics"].data_samples)
            
            if client_params:
                # Weighted average
                total_weight = sum(client_weights)
                weighted_sum = np.zeros_like(client_params[0])
                
                for param, weight in zip(client_params, client_weights):
                    weighted_sum += (weight / total_weight) * param
                
                aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def calculate_non_iid_metrics(self, round_num: int, client_updates: Dict[str, Dict]) -> NonIIDMetrics:
        """Calculate Non-IID specific metrics"""
        
        # Collect client distributions
        client_distributions = {}
        kl_scores = {}
        accuracies = []
        
        for client_id, update in client_updates.items():
            metrics = update["participation_metrics"]
            client_distributions[client_id] = metrics.data_distribution
            kl_scores[client_id] = metrics.data_heterogeneity_score
            accuracies.append(metrics.local_accuracy)
        
        # Calculate Jensen-Shannon divergence (overall heterogeneity)
        js_divergence = self._calculate_js_divergence(client_distributions)
        
        # Calculate fairness score (1 - coefficient of variation)
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)
        fairness_score = 1 - (accuracy_std / accuracy_mean) if accuracy_mean > 0 else 0
        
        # Calculate parameter diversity
        param_diversity = self._calculate_parameter_diversity(client_updates)
        
        non_iid_metrics = NonIIDMetrics(
            round_num=round_num,
            global_label_distribution=self.global_distribution,
            client_label_distributions={
                client_id: {k: v / sum(dist.values()) for k, v in dist.items()}
                for client_id, dist in client_distributions.items()
            },
            kl_divergence_scores=kl_scores,
            jensen_shannon_divergence=js_divergence,
            earth_movers_distance=0.0,  # Placeholder - complex to calculate
            accuracy_variance=accuracy_std ** 2,
            convergence_rate=0.0,  # Would need historical data
            fairness_score=fairness_score,
            parameter_diversity=param_diversity,
            consensus_measure=1 - param_diversity,  # Inverse of diversity
            aggregation_efficiency=len(client_updates) / len(self.connected_clients)
        )
        
        return non_iid_metrics
    
    def _calculate_js_divergence(self, client_distributions: Dict[str, Dict[str, int]]) -> float:
        """Calculate Jensen-Shannon divergence between client distributions"""
        if len(client_distributions) < 2:
            return 0.0
        
        # Normalize distributions
        normalized_dists = []
        for dist in client_distributions.values():
            total = sum(dist.values())
            if total > 0:
                normalized_dists.append({k: v / total for k, v in dist.items()})
        
        if len(normalized_dists) < 2:
            return 0.0
        
        # Calculate average distribution
        all_keys = set()
        for dist in normalized_dists:
            all_keys.update(dist.keys())
        
        avg_dist = {}
        for key in all_keys:
            avg_dist[key] = np.mean([dist.get(key, 0) for dist in normalized_dists])
        
        # Calculate JS divergence
        js_div = 0.0
        for dist in normalized_dists:
            kl_div = 0.0
            for key in all_keys:
                p = dist.get(key, 1e-10)
                m = avg_dist[key] + 1e-10
                if p > 0:
                    kl_div += p * np.log(p / m)
            js_div += kl_div
        
        js_div = js_div / len(normalized_dists)
        return js_div
    
    def _calculate_parameter_diversity(self, client_updates: Dict[str, Dict]) -> float:
        """Calculate parameter diversity across clients"""
        if len(client_updates) < 2:
            return 0.0
        
        # Get first layer parameters for diversity calculation
        param_name = next(iter(next(iter(client_updates.values()))["parameters"].keys()))
        
        client_params = []
        for update in client_updates.values():
            if param_name in update["parameters"]:
                params = update["parameters"][param_name].flatten()
                client_params.append(params)
        
        if len(client_params) < 2:
            return 0.0
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(client_params)):
            for j in range(i + 1, len(client_params)):
                sim = np.dot(client_params[i], client_params[j]) / (
                    np.linalg.norm(client_params[i]) * np.linalg.norm(client_params[j]) + 1e-10
                )
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity
    
    def calculate_scalability_metrics(self, round_num: int, client_updates: Dict[str, Dict], 
                                   start_time: float) -> ScalabilityMetrics:
        """Calculate scalability metrics with proper resource monitoring
        
        Args:
            round_num: Current round number
            client_updates: Dictionary of client updates for this round
            start_time: Timestamp when the round started
            
        Returns:
            ScalabilityMetrics: Calculated metrics for this round
        """
        # Get the actual number of connected clients from the configuration
        num_connected_clients = len(self.connected_clients)
        
        # If no updates received, use zeros to avoid division by zero
        if not client_updates:
            return ScalabilityMetrics(
                num_clients=num_connected_clients,
                round_num=round_num,
                average_accuracy=0.0,
                accuracy_std=0.0,
                worst_client_accuracy=0.0,
                best_client_accuracy=0.0,
                average_precision=0.0,
                precision_std=0.0,
                average_recall=0.0,
                recall_std=0.0,
                average_f1_score=0.0,
                f1_score_std=0.0,
                total_communication_time=0.0,
                average_client_latency=0.0,
                aggregation_time=0.0,
                memory_usage_mb=0.0,
                cpu_utilization=0.0,
                throughput_samples_per_sec=0.0
            )
        
        # Calculate metrics from client updates
        accuracies = [update["participation_metrics"].local_accuracy 
                     for update in client_updates.values() if update["participation_metrics"].local_accuracy is not None]
        
        # Collect enhanced ML metrics
        precisions = [update["participation_metrics"].local_precision 
                     for update in client_updates.values() if update["participation_metrics"].local_precision is not None]
        recalls = [update["participation_metrics"].local_recall 
                  for update in client_updates.values() if update["participation_metrics"].local_recall is not None]
        f1_scores = [update["participation_metrics"].local_f1_score 
                    for update in client_updates.values() if update["participation_metrics"].local_f1_score is not None]
        
        # Get system resource usage
        import psutil
        import os
        
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        # Get CPU utilization (average over last interval)
        cpu_percent = process.cpu_percent()
        
        # If CPU percent is 0 (first call), get system-wide CPU usage
        if cpu_percent == 0.0:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get GPU memory usage if available
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Note: For multi-task learning with mixed classification/regression:
        # - Classification clients report: Acc, Precision, Recall, F1
        # - Regression clients report: R² (as Acc), 1-MAE (as Precision), Pearson (as Recall), RMSE (as F1)
        # This allows aggregation while preserving task-specific semantics
        
        # Calculate metrics with proper handling of empty lists
        def safe_mean(values):
            return np.mean(values) if values else 0.0
            
        def safe_std(values):
            return np.std(values) if len(values) > 1 else 0.0
        
        scalability_metrics = ScalabilityMetrics(
            num_clients=num_connected_clients,
            round_num=round_num,
            average_accuracy=np.mean(accuracies),
            accuracy_std=np.std(accuracies),
            worst_client_accuracy=np.min(accuracies),
            best_client_accuracy=np.max(accuracies),
            average_precision=np.mean(precisions),
            precision_std=np.std(precisions),
            average_recall=np.mean(recalls),
            recall_std=np.std(recalls),
            average_f1_score=np.mean(f1_scores),
            f1_score_std=np.std(f1_scores),
            total_communication_time=time.time() - start_time,
            average_client_latency=np.mean([
                update["participation_metrics"].upload_time + update["participation_metrics"].download_time
                for update in client_updates.values()
            ]),
            aggregation_time=0.0,  # Will be updated
            memory_usage_mb=memory_usage_mb,
            cpu_utilization=cpu_percent,
            throughput_samples_per_sec=sum(
                update["participation_metrics"].data_samples for update in client_updates.values()
            ) / (time.time() - start_time)
        )
        
        logger.info(f"Resource usage - Memory: {memory_usage_mb:.1f}MB, CPU: {cpu_percent:.1f}%, GPU: {gpu_memory_mb:.1f}MB")
        
        return scalability_metrics
    
    async def run_federated_training(self):
        """Run federated training with comprehensive metrics collection"""
        
        # Wait for minimum number of clients
        while len(self.connected_clients) < self.config.min_clients:
            logger.info(f"Waiting for clients... ({len(self.connected_clients)}/{self.config.min_clients})")
            await asyncio.sleep(2)
        
        logger.info(f"Starting federated training with {len(self.connected_clients)} clients")
        
        # Training rounds
        for round_num in range(1, self.config.num_rounds + 1):
            round_start_time = time.time()
            
            logger.info(f"=== Round {round_num}/{self.config.num_rounds} ===")
            
            # For Non-LoRA, we don't actually need to send full parameters
            # Clients will train independently and send back their full parameters
            # This simulates the federated learning process without massive parameter transfer
            
            # Send lightweight training request
            training_request = {
                "type": "train",
                "round": round_num,
                "global_params": {},  # Empty for now - clients train independently
                "teacher_logits": {}  # No teacher logits for this experiment
            }
            
            logger.info(f"Sending lightweight training request for round {round_num}")
            
            # Send to all clients (create a copy to avoid dictionary change during iteration)
            connected_clients_copy = dict(self.connected_clients)
            active_clients = []
            
            for client_id, client_info in connected_clients_copy.items():
                try:
                    # Check if websocket is still open
                    websocket = client_info["websocket"]
                    if hasattr(websocket, 'closed') and websocket.closed:
                        logger.warning(f"Client {client_id} websocket is closed, removing")
                        if client_id in self.connected_clients:
                            del self.connected_clients[client_id]
                        continue
                    
                    await asyncio.wait_for(
                        client_info["websocket"].send(json.dumps(training_request)),
                        timeout=10.0  # 10 second timeout for sending
                    )
                    active_clients.append(client_id)
                    logger.info(f"Sent training request to {client_id}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout sending training request to {client_id}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                        logger.info(f"Removed timeout client {client_id}")
                        
                except (websockets.exceptions.ConnectionClosed, 
                        websockets.exceptions.WebSocketException) as e:
                    logger.error(f"WebSocket error sending to {client_id}: {e}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                        logger.info(f"Removed disconnected client {client_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to send training request to {client_id}: {e}")
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
                        logger.info(f"Removed failed client {client_id}")
            
            logger.info(f"Successfully sent requests to {len(active_clients)} clients: {active_clients}")
            
            # Check if we have enough active clients to continue
            if len(active_clients) < self.config.min_clients:
                logger.warning(f"Not enough active clients ({len(active_clients)} < {self.config.min_clients})")
                logger.warning("Skipping this round due to insufficient clients")
                continue
            
            # Wait for client updates
            logger.info("Waiting for client updates...")
            timeout_count = 0
            expected_clients = len(active_clients)  # Use the active clients we successfully sent to
            
            while (round_num not in self.client_updates or 
                   len(self.client_updates[round_num]) < expected_clients):
                await asyncio.sleep(1)
                timeout_count += 1
                
                # Log progress every 10 seconds
                if timeout_count % 10 == 0:
                    received = len(self.client_updates.get(round_num, {}))
                    logger.info(f"Round {round_num}: Received {received}/{expected_clients} client updates")
                
                if timeout_count > self.config.timeout:
                    logger.warning(f"Timeout waiting for clients in round {round_num}")
                    logger.warning(f"Expected {expected_clients}, received {len(self.client_updates.get(round_num, {}))}")
                    break
            
            # Process round results
            if round_num in self.client_updates:
                aggregation_start = time.time()
                
                # Aggregate parameters
                aggregated_params = self.aggregate_parameters(self.client_updates[round_num])
                
                # Update global model (for metrics collection, we'll simulate this)
                if aggregated_params:
                    logger.info(f"Simulated global model update with {len(aggregated_params)} parameter groups")
                    # In a real scenario, we would update the global model here
                    # For metrics collection, we'll skip the actual update to avoid memory issues
                
                aggregation_time = time.time() - aggregation_start
                
                # Calculate metrics
                non_iid_metrics = self.calculate_non_iid_metrics(round_num, self.client_updates[round_num])
                scalability_metrics = self.calculate_scalability_metrics(
                    round_num, self.client_updates[round_num], round_start_time
                )
                scalability_metrics.aggregation_time = aggregation_time
                
                # Store metrics
                self.non_iid_metrics_history.append(non_iid_metrics)
                self.scalability_metrics_history.append(scalability_metrics)
                
                logger.info(f"Round {round_num} completed in {time.time() - round_start_time:.2f}s")
                logger.info(f"Clients: {len(self.client_updates[round_num])}, "
                           f"Avg Accuracy: {scalability_metrics.average_accuracy:.4f}, "
                           f"JS Divergence: {non_iid_metrics.jensen_shannon_divergence:.4f}")
        
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
        
        logger.info("Federated training completed")
    
    def save_metrics_to_csv(self):
        """Save all metrics to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory with experiment-specific naming
        experiment_id = f"{self.config.max_clients}c_{self.config.num_rounds}r_{timestamp}"
        results_dir = Path("no_lora_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save participation metrics with experiment-specific naming
        participation_file = results_dir / f"participation_metrics_{self.config.max_clients}clients_{timestamp}.csv"
        with open(participation_file, 'w', newline='') as f:
            if self.participation_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.participation_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.participation_metrics_history:
                    row = asdict(metrics)
                    # Convert dict fields to JSON strings for CSV
                    row['data_distribution'] = json.dumps(row['data_distribution'])
                    writer.writerow(row)
        
        # Save Non-IID metrics with experiment-specific naming
        non_iid_file = results_dir / f"non_iid_metrics_{self.config.max_clients}clients_{timestamp}.csv"
        with open(non_iid_file, 'w', newline='') as f:
            if self.non_iid_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.non_iid_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.non_iid_metrics_history:
                    row = asdict(metrics)
                    # Convert dict fields to JSON strings for CSV
                    for field in ['global_label_distribution', 'client_label_distributions', 'kl_divergence_scores']:
                        row[field] = json.dumps(row[field])
                    writer.writerow(row)
        
        # Save scalability metrics with experiment-specific naming
        scalability_file = results_dir / f"scalability_metrics_{self.config.max_clients}clients_{timestamp}.csv"
        with open(scalability_file, 'w', newline='') as f:
            if self.scalability_metrics_history:
                writer = csv.DictWriter(f, fieldnames=asdict(self.scalability_metrics_history[0]).keys())
                writer.writeheader()
                for metrics in self.scalability_metrics_history:
                    writer.writerow(asdict(metrics))
        
        logger.info(f"Metrics saved to CSV files in {results_dir}")
        logger.info(f"Files: {participation_file.name}, {non_iid_file.name}, {scalability_file.name}")
    
    async def start_server(self):
        """Start the federated learning server"""
        logger.info(f"Starting server on port {self.config.port}")
        
        # Start WebSocket server with proper configuration
        server = await websockets.serve(
            self.client_handler,
            "localhost",
            self.config.port,
            ping_interval=None,  # Disable automatic ping
            ping_timeout=None,   # Disable ping timeout
            close_timeout=15,    # Wait 15 seconds for close
            max_size=None,       # No message size limit
            compression=None     # Disable compression for stability
        )
        
        logger.info(f"Server listening on localhost:{self.config.port}")
        
        # Run federated training
        await self.run_federated_training()
        
        server.close()
        await server.wait_closed()

async def run_no_lora_experiment(config: NoLoRAConfig, mode: str, client_id: str = None, 
                               task: str = None, total_clients: int = 5):
    """Run non-LoRA federated learning experiment"""
    
    if mode == "server":
        server = NoLoRAFederatedServer(config)
        await server.start_server()
    
    elif mode == "client":
        if not client_id or not task:
            raise ValueError("Client mode requires client_id and task")
        
        client = NoLoRAFederatedClient(client_id, task, config, total_clients)
        await client.run_client("localhost", config.port)
    
    else:
        raise ValueError("Mode must be 'server' or 'client'")

def main():
    parser = argparse.ArgumentParser(description="Non-LoRA Federated Learning System")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--task", choices=["sst2", "qqp", "stsb"], help="Task name (required for client mode)")
    parser.add_argument("--port", type=int, default=8771, help="Server port")
    parser.add_argument("--rounds", type=int, default=22, help="Number of federated rounds")
    parser.add_argument("--samples", type=int, help="Data samples per client (overrides config file)")
    parser.add_argument("--total_clients", type=int, help="Total number of clients (overrides config file)")
    parser.add_argument("--distribution", choices=["iid", "non_iid", "pathological"], 
                       help="Data distribution type (overrides config file)")
    parser.add_argument("--alpha", type=float, help="Non-IID alpha parameter (overrides config file)")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Load config from file
    config_parser = configparser.ConfigParser()
    config_parser.read('experiment_config.ini')
    
    # Get default values from config file if not provided in command line
    if args.mode == "server":
        section = 'SCENARIO_1'  # or get this from command line if needed
        if not args.samples:
            args.samples = int(config_parser.get(section, 'samples_per_client', fallback=1000))
        if not args.total_clients:
            args.total_clients = int(config_parser.get(section, 'num_clients', fallback=5))
        if not args.distribution:
            args.distribution = config_parser.get(section, 'data_distribution', fallback='non_iid')
        if not args.alpha:
            args.alpha = float(config_parser.get(section, 'non_iid_alpha', fallback=0.5))
    
    logger.info(f"Using configuration: {args}")
    
    # Create configuration
    config = NoLoRAConfig(
        port=args.port,
        num_rounds=args.rounds,
        data_samples_per_client=args.samples,
        data_distribution=args.distribution,
        non_iid_alpha=args.alpha,
        max_clients=args.total_clients
    )
    
    # Run experiment
    asyncio.run(run_no_lora_experiment(
        config, args.mode, args.client_id, args.task, args.total_clients
    ))

if __name__ == "__main__":
    main()
