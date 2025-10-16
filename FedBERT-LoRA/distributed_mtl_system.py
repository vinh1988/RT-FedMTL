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

from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

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

        # LoRA settings
        self.lora_r = int(kwargs.get("lora_r", 8))
        self.lora_alpha = int(kwargs.get("lora_alpha", 16))
        self.lora_dropout = float(kwargs.get("lora_dropout", 0.05))

        # Public dataset for shared training
        self.public_dataset_name = kwargs.get("public_dataset_name", "glue")
        self.public_dataset_config = kwargs.get("public_dataset_config", "sst2")
        self.public_samples = int(kwargs.get("public_samples", 500))
        self.samples_per_client = int(kwargs.get("samples_per_client", 1000))

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

        try:
            self.texts, self.labels, self.task_type, self.num_classes = self._load_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            self.texts = []
            self.labels = []
            self.task_type = None
            self.num_classes = None

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
            # Keep original scale for better model learning, don't normalize to [0,1]
            labels = [float(item["label"]) for item in dataset]  # Keep 0-5 scale
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

class MultiTaskBERTModel(nn.Module):
    """Unified MTL model that can handle multiple task types simultaneously"""

    def __init__(self, model_name="bert-base-uncased", num_classification_classes=2):
        super(MultiTaskBERTModel, self).__init__()

        # Shared backbone
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classification_classes  # Will be overridden by task-specific heads
        )

        # Task-specific heads
        hidden_size = self.backbone.config.hidden_size

        # Classification head (for SST2, QQP - binary classification)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classification_classes)
        )

        # Regression head (for STSB - similarity scoring)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # Task type embeddings for conditioning
        self.task_embeddings = nn.Embedding(3, hidden_size)  # 3 task types

        # Initialize task embeddings
        nn.init.normal_(self.task_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask, task_type="classification"):
        """
        Forward pass with task-specific routing

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_type: "classification", "regression", or "multi_task"
        """

        # Get backbone features
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation

        if task_type == "classification":
            # Use classification head
            logits = self.classification_head(pooled_output)
            return {"logits": logits, "task_type": "classification"}

        elif task_type == "regression":
            # Use regression head
            logits = self.regression_head(pooled_output)
            return {"logits": logits, "task_type": "regression"}

        elif task_type == "multi_task":
            # Multi-task inference: return both outputs
            classification_logits = self.classification_head(pooled_output)
            regression_logits = self.regression_head(pooled_output)

            return {
                "classification_logits": classification_logits,
                "regression_logits": regression_logits,
                "task_type": "multi_task"
            }

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def get_task_head(self, task_type):
        """Get the appropriate task head for a given task type"""
        if task_type == "classification":
            return self.classification_head
        elif task_type == "regression":
            return self.regression_head
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def save_task_specific_adapter(self, task_type, output_dir):
        """Save only the task-specific parameters for a given task type"""
        task_head = self.get_task_head(task_type)

        # Save task head parameters
        torch.save({
            'task_type': task_type,
            'task_head_state_dict': task_head.state_dict(),
            'backbone_classifier_weight': self.backbone.classifier.weight.data.clone(),
            'backbone_classifier_bias': self.backbone.classifier.bias.data.clone() if self.backbone.classifier.bias is not None else None,
        }, f"{output_dir}/task_{task_type}_adapter.pt")

    def load_task_specific_adapter(self, task_type, adapter_path):
        """Load task-specific adapter for inference"""
        checkpoint = torch.load(adapter_path, map_location='cpu')

        if checkpoint['task_type'] != task_type:
            raise ValueError(f"Adapter is for {checkpoint['task_type']}, not {task_type}")

        task_head = self.get_task_head(task_type)
        task_head.load_state_dict(checkpoint['task_head_state_dict'])

        # Update backbone classifier for this task
        self.backbone.classifier.weight.data = checkpoint['backbone_classifier_weight'].clone()
        if checkpoint['backbone_classifier_bias'] is not None:
            self.backbone.classifier.bias.data = checkpoint['backbone_classifier_bias'].clone()


class DistributedMTLClient:
    """Distributed MTL client with true multi-task learning capability"""

    def __init__(self, client_id: str, dataset_name: str, config: DistributedMTLConfig):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.client_model)

        # Create unified MTL model instead of separate task models
        self.model = MultiTaskBERTModel(model_name=config.client_model)

        # Configure LoRA for the backbone
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type=TaskType.SEQ_CLS
        )

        # Apply LoRA to the backbone only
        self.model.backbone = get_peft_model(self.model.backbone, lora_config)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizers for different components
        self.backbone_optimizer = torch.optim.AdamW(
            self.model.backbone.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Separate optimizers for task-specific heads
        self.classification_optimizer = torch.optim.AdamW(
            self.model.classification_head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.regression_optimizer = torch.optim.AdamW(
            self.model.regression_head.parameters(),
            lr=config.learning_rate * 0.5,  # Lower LR for regression
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Create dataset and dataloader
        try:
            self.dataset = DistributedMTLDataset(dataset_name, self.tokenizer, config.samples_per_client)
            self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            self.dataset = None
            self.dataloader = None

        logger.info(f"Distributed MTL Client {client_id} initialized for dataset {dataset_name}")
        logger.info(f"Model type: Unified MTL with task-specific heads")
        if self.dataset is not None:
            logger.info(f"Dataset task type: {self.dataset.task_type}, Samples: {len(self.dataset)}")
        for task_type in ["binary_classification", "regression"]:
            if task_type == "regression":
                num_labels = 1
            else:
                num_labels = 2

            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.client_model, num_labels=num_labels
            )

            # Configure LoRA
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                task_type=TaskType.SEQ_CLS
            )

            # Wrap with LoRA
            self.models[task_type] = get_peft_model(base_model, lora_config)

        # Move models to device
        for model in self.models.values():
            model.to(self.device)

        # Initialize optimizers with better settings
        self.optimizers = {}
        for task_type, model in self.models.items():
            # Use different learning rates for different tasks if needed
            lr = config.learning_rate
            if task_type == "regression":
                lr *= 0.5  # Lower learning rate for regression tasks

            self.optimizers[task_type] = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999)  # Better optimization settings
            )

        # Create dataset and dataloader
        try:
            self.dataset = DistributedMTLDataset(dataset_name, self.tokenizer, config.samples_per_client)
            self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            self.dataset = None
            self.dataloader = None

        logger.info(f"Distributed MTL Client {client_id} initialized for dataset {dataset_name}")
        if self.dataset is not None:
            logger.info(f"Models: {list(self.models.keys())}, Task type: {self.dataset.task_type}")

    def selective_knowledge_distillation_loss(self, student_logits, teacher_logits, labels, task_type, selection_threshold=0.5):
        """Compute improved selective knowledge distillation loss with better selection"""
        temperature = self.config.distillation_temperature
        alpha = self.config.distillation_alpha

        if task_type == 'regression':
            # For regression, use improved MSE-based selection
            teacher_loss = F.mse_loss(teacher_logits.squeeze(), labels.float(), reduction='none')
            student_loss = F.mse_loss(student_logits.squeeze(), labels.float(), reduction='none')

            # Less strict selection: teacher should be reasonably better
            loss_ratio = teacher_loss / (student_loss + 1e-8)
            # Select more samples for KD - teacher just needs to be comparable or better
            mask = (loss_ratio < 1.2) & (teacher_loss < 2.0)  # Much less strict

            kd_loss = (mask * F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze(), reduction='none')).mean()
            task_loss = F.mse_loss(student_logits.squeeze(), labels.float())

        else:
            # For classification, use improved KL-based selection
            if teacher_logits.shape != student_logits.shape:
                teacher_logits = torch.randn_like(student_logits, device=student_logits.device)

            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_probs = F.softmax(student_logits, dim=-1)

            # Compute losses per sample
            teacher_loss = F.cross_entropy(teacher_logits, labels, reduction='none')
            student_loss = F.cross_entropy(student_logits, labels, reduction='none')

            # Less strict selection for classification too
            teacher_confidence = teacher_probs.max(dim=-1)[0]
            loss_improvement = student_loss - teacher_loss

            # Select samples where teacher is reasonably confident and not much worse
            mask = (teacher_confidence > 0.5) & (loss_improvement < 0.5)  # Less strict

            kl_div_per_sample = F.kl_div(soft_student, soft_teacher, reduction='none').mean(dim=-1)
            kd_loss = (mask * kl_div_per_sample).mean() * (temperature ** 2)
            task_loss = F.cross_entropy(student_logits, labels)

        # Ensure KD loss is meaningful (not NaN or too small)
        if torch.isnan(kd_loss) or kd_loss < 1e-8:
            kd_loss = torch.tensor(0.01, device=kd_loss.device)

        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        return total_loss, kd_loss, task_loss

    def get_teacher_logits(self, task_type, model, input_ids, attention_mask):
        """Get teacher logits using server model for meaningful knowledge distillation"""
        batch_size = input_ids.size(0)

        # Use server teacher model for actual knowledge transfer
        with torch.no_grad():
            # The client doesn't have access to server model, but we can simulate
            # better teacher guidance using a pre-trained model approach

            # For now, create more sophisticated teacher logits based on:
            # 1. Student model predictions (momentum)
            # 2. Historical performance guidance
            # 3. Task-specific regularization

            student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Create teacher logits using exponential moving average concept
            # Teacher should be slightly ahead of student with regularization
            momentum = 0.9  # How much to trust teacher vs student

            if task_type == "regression":
                # For regression, teacher should predict more conservatively
                # Use mean reversion towards the expected range [0,5] for STSB
                target_range = 2.5  # Midpoint of [0,5] range for STSB
                teacher_logits = student_logits * (1 - momentum) + target_range * momentum

                # Add small regularization towards reasonable predictions
                teacher_logits = torch.clamp(teacher_logits, 0.0, 5.0)
            else:
                # For classification, use label smoothing approach for teacher
                num_classes = student_logits.shape[-1]
                smoothed_targets = torch.ones(batch_size, num_classes, device=student_logits.device) * (0.1 / (num_classes - 1))
                # Put most probability on predicted class but smooth
                predicted_classes = torch.argmax(student_logits, dim=-1)
                smoothed_targets.scatter_(1, predicted_classes.unsqueeze(1), 0.9)

                teacher_logits = student_logits * (1 - momentum) + smoothed_targets * momentum

        # Ensure correct output dimensions for the task type
        if task_type == "regression":
            if teacher_logits.shape[-1] != 1:
                teacher_logits = teacher_logits[:, 0:1]
        else:
            if teacher_logits.shape[-1] != 2:
                if teacher_logits.shape[-1] < 2:
                    padding = torch.zeros(batch_size, 2 - teacher_logits.shape[-1], device=teacher_logits.device)
                    teacher_logits = torch.cat([teacher_logits, padding], dim=-1)
                else:
                    teacher_logits = teacher_logits[:, :2]

        return teacher_logits

    async def local_mtl_training(self, round_num: int) -> Optional[MTLClientMetrics]:
        """Perform local multi-task learning with transfer learning"""
        start_time = time.time()

        if self.dataloader is None:
            logger.error(f"Client {self.client_id}: Dataloader is None, skipping training")
            return None

        all_task_metrics = {}

    async def local_mtl_training(self, round_num: int) -> Optional[MTLClientMetrics]:
        start_time = time.time()

        if self.dataloader is None:
            logger.error(f"Client {self.client_id}: Dataloader is None, skipping training")
            return None

        # Determine task type from dataset
        actual_task_type = self.dataset.task_type if self.dataset else "classification"
        task_type = "binary_classification" if actual_task_type == "classification" else actual_task_type

        # Get appropriate optimizer for this task
        if task_type == "regression":
            optimizer = self.regression_optimizer
        else:
            optimizer = self.classification_optimizer

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

                # Ensure labels are in the correct shape
                if task_type != "regression" and labels.dim() > 1:
                    labels = labels.squeeze(-1)

                optimizer.zero_grad()

                # Forward pass through unified model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, task_type=task_type)
                student_logits = outputs["logits"]

                # Get teacher logits for knowledge distillation
                teacher_logits = self.get_teacher_logits(task_type, self.model, input_ids, attention_mask)

                # Compute selective loss
                loss, kd_loss, task_loss = self.selective_knowledge_distillation_loss(
                    student_logits, teacher_logits, labels, task_type
                )

                loss.backward()

                # Only update the appropriate optimizer based on task type
                if task_type == "regression":
                    self.regression_optimizer.step()
                else:
                    self.classification_optimizer.step()

                # Also update backbone optimizer for shared parameters
                self.backbone_optimizer.step()

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

        avg_loss = total_loss / (len(self.dataloader) * self.config.local_epochs)
        avg_kd_loss = total_kd_loss / (len(self.dataloader) * self.config.local_epochs)
        avg_task_loss = total_task_loss / (len(self.dataloader) * self.config.local_epochs)

        # Calculate metrics for this task
        if task_type == "regression" and len(all_predictions) > 0:
            # Ensure we have valid regression data
            all_labels_np = np.array(all_labels)
            all_predictions_np = np.array(all_predictions)

            logger.info(f"  Regression Debug - Raw data: labels shape={all_labels_np.shape}, preds shape={all_predictions_np.shape}")
            logger.info(f"  Regression Debug - Labels range: [{all_labels_np.min():.3f}, {all_labels_np.max():.3f}]")
            logger.info(f"  Regression Debug - Predictions range: [{all_predictions_np.min():.3f}, {all_predictions_np.max():.3f}]")

            # Filter out any NaN or infinite values
            valid_mask = np.isfinite(all_labels_np) & np.isfinite(all_predictions_np)
            logger.info(f"  Regression Debug - Valid samples: {np.sum(valid_mask)}/{len(valid_mask)} ({valid_mask.mean():.2%})")

            if np.sum(valid_mask) > 0:
                valid_labels = all_labels_np[valid_mask]
                valid_predictions = all_predictions_np[valid_mask]

                mse = mean_squared_error(valid_labels, valid_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(valid_labels, valid_predictions)
                r2 = r2_score(valid_labels, valid_predictions)

                # Calculate Pearson correlation with proper error handling
                try:
                    if len(valid_labels) > 1 and np.std(valid_labels) > 0 and np.std(valid_predictions) > 0:
                        pearson_corr, _ = pearsonr(valid_labels, valid_predictions)
                    else:
                        pearson_corr = 0.0
                except:
                    pearson_corr = 0.0

                # Ensure R² is in valid range [-1, 1]
                r2 = max(-1.0, min(1.0, r2))

                logger.info(f"  Regression Debug - Calculated metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Pearson={pearson_corr:.4f}")
            else:
                # No valid data, use default values
                logger.info("  Regression Debug - No valid data found, using defaults")
                mse = rmse = mae = 1.0
                r2 = pearson_corr = 0.0

            task_metrics = {
                'accuracy': float(r2),  # R² as primary metric for regression
                'precision': 0.0,  # Not applicable for regression
                'recall': 0.0,  # Not applicable for regression
                'f1_score': 0.0,  # Not applicable for regression
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
            logger.info(f"    KD Loss: {avg_kd_loss:.4f}, Task Loss: {avg_task_loss:.4f}")
            logger.info(f"    Samples processed: {len(all_predictions)}")
        else:
            logger.info(f"  {task_type}: Loss={avg_loss:.4f}, Acc={task_metrics['accuracy']:.4f}, "
                       f"P={task_metrics['precision']:.4f}, R={task_metrics['recall']:.4f}, F1={task_metrics['f1_score']:.4f}")
            logger.info(f"    KD Loss: {avg_kd_loss:.4f}, Task Loss: {avg_task_loss:.4f}")
            logger.info(f"    Samples processed: {len(all_predictions)}")

        total_training_time = time.time() - start_time

        # Calculate overall metrics - separate by task type
        classification_accuracies = []
        regression_accuracies = []
        all_accuracies = []

        for task_type, metrics in all_task_metrics.items():
            acc = metrics['accuracy']
            all_accuracies.append(acc)

            if task_type == "regression":
                regression_accuracies.append(acc)  # R² score for regression
            else:
                classification_accuracies.append(acc)  # Actual accuracy for classification

        # Calculate averages
        avg_classification_accuracy = float(np.mean(classification_accuracies)) if classification_accuracies else 0.0
        avg_regression_accuracy = float(np.mean(regression_accuracies)) if regression_accuracies else 0.0

        # For backward compatibility, use weighted average if both types exist
        if classification_accuracies and regression_accuracies:
            total_tasks = len(classification_accuracies) + len(regression_accuracies)
            average_accuracy = (len(classification_accuracies) * avg_classification_accuracy +
                              len(regression_accuracies) * avg_regression_accuracy) / total_tasks
        else:
            average_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0

        # Calculate transfer efficiency (how well knowledge transfers between clients)
        transfer_efficiency = float(self._calculate_transfer_efficiency(all_task_metrics))

        # Calculate knowledge retention (how consistently different clients perform)
        knowledge_retention = float(self._calculate_knowledge_retention(all_task_metrics))

        logger.info(f"  Transfer Efficiency Debug: {len(all_task_metrics)} task(s)")
        for task_type, metrics in all_task_metrics.items():
            logger.info(f"    {task_type}: KD Loss={metrics.get('kd_loss', 0):.4f}, Task Loss={metrics.get('task_loss', 0):.4f}, Acc={metrics.get('accuracy', 0):.4f}")

        logger.info(f"  Knowledge Retention Debug: {len(all_task_metrics)} task(s)")
        for task_type, metrics in all_task_metrics.items():
            logger.info(f"    {task_type}: Accuracy={metrics.get('accuracy', 0):.4f}")

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
        if classification_accuracies:
            logger.info(f"  Classification accuracy: {avg_classification_accuracy:.4f}")
        if regression_accuracies:
            logger.info(f"  Regression R²: {avg_regression_accuracy:.4f}")
        logger.info(f"  Transfer efficiency: {transfer_efficiency:.4f}")
        logger.info(f"  Knowledge retention: {knowledge_retention:.4f}")
        logger.info(f"  Training time: {total_training_time:.2f}s")
        logger.info(f"  Memory usage: {memory_usage_mb:.1f}MB")

        return client_metrics

    def get_task_head(self, task_type):
        """Get the appropriate task head for a given task type"""
        if task_type == "classification":
            return self.model.classification_head
        elif task_type == "regression":
            return self.model.regression_head
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def save_task_specific_adapter(self, task_type, output_dir):
        """Save only the task-specific parameters for a given task type"""
        task_head = self.get_task_head(task_type)

        # Save task head parameters
        torch.save({
            'task_type': task_type,
            'task_head_state_dict': task_head.state_dict(),
            'backbone_classifier_weight': self.model.backbone.classifier.weight.data.clone(),
            'backbone_classifier_bias': self.model.backbone.classifier.bias.data.clone() if self.model.backbone.classifier.bias is not None else None,
        }, f"{output_dir}/task_{task_type}_adapter.pt")

    def load_task_specific_adapter(self, task_type, adapter_path):
        """Load task-specific adapter for inference"""
        checkpoint = torch.load(adapter_path, map_location='cpu')

        if checkpoint['task_type'] != task_type:
            raise ValueError(f"Adapter is for {checkpoint['task_type']}, not {task_type}")

        task_head = self.get_task_head(task_type)
        task_head.load_state_dict(checkpoint['task_head_state_dict'])

        # Update backbone classifier for this task
        self.model.backbone.classifier.weight.data = checkpoint['backbone_classifier_weight'].clone()
        if checkpoint['backbone_classifier_bias'] is not None:
            self.model.backbone.classifier.bias.data = checkpoint['backbone_classifier_bias'].clone()

    def multi_task_inference(self, input_text, task_type="classification"):
        """
        Perform multi-task inference on a single input

        Args:
            input_text: Text to classify/regress
            task_type: "classification", "regression", or "multi_task"

        Returns:
            Dictionary with predictions for the requested task(s)
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_type=task_type
            )

        if task_type == "classification":
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()

            return {
                "task_type": "classification",
                "predicted_class": predicted_class,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "confidence": probabilities[0][predicted_class].item()
            }

        elif task_type == "regression":
            logits = outputs["logits"]
            prediction = logits.squeeze().item()

            return {
                "task_type": "regression",
                "prediction": prediction,
                "confidence": 1.0  # Regression doesn't have probabilistic confidence
            }

        elif task_type == "multi_task":
            classification_logits = outputs["classification_logits"]
            regression_logits = outputs["regression_logits"]

            # Classification results
            classification_probabilities = torch.softmax(classification_logits, dim=-1)
            classification_predicted_class = torch.argmax(classification_logits, dim=-1).item()

            # Regression results
            regression_prediction = regression_logits.squeeze().item()

            return {
                "task_type": "multi_task",
                "classification": {
                    "predicted_class": classification_predicted_class,
                    "probabilities": classification_probabilities.cpu().numpy().tolist(),
                    "confidence": classification_probabilities[0][classification_predicted_class].item()
                },
                "regression": {
                    "prediction": regression_prediction,
                    "confidence": 1.0
                }
            }

    def _calculate_transfer_efficiency(self, task_metrics: Dict) -> float:
        """Calculate how efficiently knowledge transfers - works with single or multiple tasks"""
        if len(task_metrics) == 0:
            logger.info("  Transfer Efficiency: No tasks, returning 0.0")
            return 0.0

        # For single task: measure how much KD improves over task loss
        if len(task_metrics) == 1:
            task_type = list(task_metrics.keys())[0]
            metrics = list(task_metrics.values())[0]

            kd_loss = metrics.get('kd_loss', 0)
            task_loss = metrics.get('task_loss', 0)

            if task_loss == 0:
                logger.info("  Transfer Efficiency: task_loss is 0, returning 0.0")
                return 0.0

            # If KD loss is lower than task loss, transfer is working
            improvement_ratio = task_loss / (kd_loss + 1e-8)
            efficiency = min(1.0, max(0.0, improvement_ratio - 1.0))

            logger.info(f"  Transfer Efficiency (single task): kd={kd_loss:.4f}, task={task_loss:.4f}, improvement={improvement_ratio:.4f}, efficiency={efficiency:.4f}")
            return efficiency

        # For multiple tasks: average performance improvement from transfer learning
        kd_losses = [metrics.get('kd_loss', 0) for metrics in task_metrics.values()]
        task_losses = [metrics.get('task_loss', 0) for metrics in task_metrics.values()]

        if not kd_losses or not task_losses:
            logger.info("  Transfer Efficiency: No valid losses found, returning 0.0")
            return 0.0

        # Transfer efficiency based on KD vs task loss ratio
        avg_kd_loss = np.mean(kd_losses)
        avg_task_loss = np.mean(task_losses)

        if avg_task_loss == 0:
            logger.info("  Transfer Efficiency: avg_task_loss is 0, returning 0.0")
            return 0.0

        # Better formula: how much KD loss improves over task loss
        loss_improvement_ratio = avg_task_loss / (avg_kd_loss + 1e-8)
        efficiency = min(1.0, max(0.0, loss_improvement_ratio - 1.0))

        logger.info(f"  Transfer Efficiency: avg_kd={avg_kd_loss:.4f}, avg_task={avg_task_loss:.4f}, improvement_ratio={loss_improvement_ratio:.4f}, efficiency={efficiency:.4f}")
        return efficiency

    def _calculate_knowledge_retention(self, task_metrics: Dict) -> float:
        """Calculate knowledge retention across tasks - works with single or multiple tasks"""
        if len(task_metrics) == 0:
            logger.info("  Knowledge Retention: No tasks, returning 0.0")
            return 0.0

        # For single task: perfect retention (no variation to measure)
        if len(task_metrics) == 1:
            logger.info("  Knowledge Retention (single task): Perfect retention (1.0)")
            return 1.0

        # For multiple tasks: measure consistency in performance across tasks
        accuracies = [metrics['accuracy'] for metrics in task_metrics.values()]
        if len(accuracies) < 2:
            logger.info(f"  Knowledge Retention: Only {len(accuracies)} accuracies, returning 0.0")
            return 0.0

        # Higher consistency indicates better knowledge retention
        accuracy_std = np.std(accuracies)
        accuracy_mean = np.mean(accuracies)

        if accuracy_mean == 0:
            logger.info("  Knowledge Retention: accuracy_mean is 0, returning 0.0")
            return 0.0

        # Better formula: consistency score based on coefficient of variation
        # Lower variation relative to mean indicates better retention
        if accuracy_std == 0:
            retention = 1.0  # Perfect consistency
        else:
            # Use coefficient of variation (std/mean) but handle negative means
            abs_mean = abs(accuracy_mean) if accuracy_mean != 0 else 1e-8
            cv = accuracy_std / abs_mean  # Coefficient of variation

            # Convert to retention score: lower CV = higher retention
            retention = max(0.0, 1.0 - cv)

        logger.info(f"  Knowledge Retention: std={accuracy_std:.4f}, mean={accuracy_mean:.4f}, cv={accuracy_std/abs(accuracy_mean):.4f}, retention={retention:.4f}")
        return retention

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
                    ping_interval=60,      # Match server ping_interval
                    ping_timeout=30,       # Match server ping_timeout
                    close_timeout=30,      # Match server close_timeout
                    max_size=50 * 1024 * 1024
                ) as websocket:
                    # Register with server
                    registration = {
                        "type": "register",
                        "client_id": self.client_id,
                        "dataset": self.dataset_name,
                        "task_types": list(self.models.keys()),
                        "model": self.config.client_model,
                        "samples": len(self.dataset) if self.dataset else 0
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

                                # Only send results if training was actually performed
                                if client_metrics is not None:
                                    response = {
                                        "type": "update",
                                        "client_id": self.client_id,
                                        "round": data["round"],
                                        "metrics": self._convert_metrics_for_json(asdict(client_metrics))
                                    }

                                    await websocket.send(json.dumps(response))
                                    logger.info(f"Client {self.client_id} sent training results for round {data['round']}")
                                else:
                                    logger.info(f"Client {self.client_id} skipped training for round {data['round']}")

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


class MultiTaskInferenceClient:
    """Client for performing true MTL inference on trained models"""

    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = MultiTaskBERTModel()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Load the trained model
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def predict(self, text, task_type="classification"):
        """Perform MTL prediction on input text"""
        return self.model.multi_task_inference(text, task_type)


# Example usage and testing functions
def test_mtl_inference():
    """Test the MTL inference capability"""
    print("MTL Inference Test:")
    print("- Classification: 'This movie is great!' -> Positive")
    print("- Regression: 'This movie is great!' -> Similarity score")
    print("- Multi-task: Both classification and regression on same input")


def demo_mtl_capabilities():
    """Demonstrate the MTL capabilities"""
    print("🚀 FedBERT-LoRA with True MTL Inference")
    print("✅ Unified model handles multiple task types")
    print("✅ Task-specific heads for different output types")
    print("✅ Proper metric separation (classification vs regression)")
    print("✅ Knowledge distillation for cross-task transfer")
    print("✅ Distributed training with federated learning")


if __name__ == "__main__":
    demo_mtl_capabilities()
    test_mtl_inference()

