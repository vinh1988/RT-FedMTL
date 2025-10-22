#!/usr/bin/env python3
"""
Local Multi-Task Trainer with LoRA and Knowledge Distillation
Standalone training (Federated Learning components removed)
"""

import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class LocalTrainer:
    """Local trainer for multi-task learning with LoRA and KD"""

    def __init__(self, student_model, kd_manager, dataset_handlers, config):
        self.student_model = student_model
        self.kd_manager = kd_manager
        self.dataset_handlers = dataset_handlers
        self.config = config
        self.device = self._get_device()

        # Setup training components
        self.setup_training()

    def _get_device(self):
        """Get available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Move model to device first
        self.student_model.to(self.device)

        # Setup optimizer for LoRA parameters only
        lora_params = []
        for name, param in self.student_model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append(param)

        self.optimizer = torch.optim.AdamW(lora_params, lr=self.config.learning_rate)

        # Setup scheduler
        total_steps = len(self.dataset_handlers) * self.config.local_epochs * 100  # Approximate
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Setup loss functions
        self.task_criterion = torch.nn.CrossEntropyLoss()
        self.mse_criterion = torch.nn.MSELoss()

    def prepare_batch(self, batch, task):
        """Prepare batch for training"""
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Handle different task types
        if task in ['sst2', 'qqp']:
            # Classification tasks
            labels = labels.long()
        # STSB is regression, labels are already float

        return input_ids, attention_mask, labels

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch across all tasks"""
        self.student_model.train()
        total_loss = 0
        task_losses = {}

        for task, handler in self.dataset_handlers.items():
            task_loss = 0
            num_batches = 0

            # Get dataloader for this task
            dataloader = handler.get_dataloader()

            for batch in dataloader:
                input_ids, attention_mask, labels = self.prepare_batch(batch, task)

                # Forward pass through student model
                student_outputs = self.student_model(input_ids, attention_mask, task)

                # Calculate task-specific loss
                if task in ['sst2', 'qqp']:
                    task_loss_batch = self.task_criterion(student_outputs, labels)
                else:  # STSB regression
                    task_loss_batch = self.mse_criterion(student_outputs.squeeze(), labels.float())

                # Add KD loss if teacher model is available
                if self.kd_manager.teacher_model is not None:
                    # Get teacher outputs for KD (ensure same device as student)
                    with torch.no_grad():
                        teacher_outputs = self.kd_manager.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

                    # Ensure teacher outputs are on the same device as student outputs
                    if teacher_outputs.device != student_outputs.device:
                        teacher_outputs = teacher_outputs.to(student_outputs.device)

                    # Calculate KD loss
                    kd_loss = self.kd_manager.teacher_to_student_kd_loss(
                        student_outputs, teacher_outputs, labels,
                        "regression" if task == "stsb" else "classification"
                    )
                    total_loss_batch = (1 - self.config.kd_alpha) * task_loss_batch + self.config.kd_alpha * kd_loss
                else:
                    total_loss_batch = task_loss_batch

                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                self.scheduler.step()

                task_loss += total_loss_batch.item()
                total_loss += total_loss_batch.item()
                num_batches += 1

            task_losses[task] = task_loss / num_batches if num_batches > 0 else 0

        return {
            'total_loss': total_loss / sum(len(handler.get_dataloader()) for handler in self.dataset_handlers.values()),
            **task_losses
        }

    def evaluate_task(self, task: str, handler) -> Dict[str, float]:
        """Evaluate performance on a specific task"""
        self.student_model.eval()
        dataloader = handler.get_dataloader()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = self.prepare_batch(batch, task)
                outputs = self.student_model(input_ids, attention_mask, task)

                if task in ['sst2', 'qqp']:
                    # Classification
                    loss = self.task_criterion(outputs, labels)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                else:
                    # Regression (STSB)
                    loss = self.mse_criterion(outputs.squeeze(), labels.float())
                    # For regression, we'll use MSE as the primary metric

                total_loss += loss.item()
                total += labels.size(0)

        metrics = {'loss': total_loss / len(dataloader)}
        if task in ['sst2', 'qqp']:
            metrics['accuracy'] = correct / total
        else:
            metrics['mse'] = total_loss / len(dataloader)

        return metrics

    def train_local_mtl(self, num_rounds: int) -> Dict[str, Dict[str, float]]:
        """Train locally with multi-task learning"""
        logger.info(f"Starting local MTL training for {num_rounds} rounds")

        results = {}

        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")

            # Train for one epoch
            train_metrics = self.train_epoch(round_num)

            # Evaluate on all tasks
            round_results = {}
            for task, handler in self.dataset_handlers.items():
                eval_metrics = self.evaluate_task(task, handler)
                round_results[task] = eval_metrics

            results[f"round_{round_num + 1}"] = round_results

            logger.info(f"Round {round_num + 1} completed. Train loss: {train_metrics['total_loss']:.4f}")

        return results

    def save_model(self, path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'config': self.config,
            'tasks': list(self.dataset_handlers.keys())
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
