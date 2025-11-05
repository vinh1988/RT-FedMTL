import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

class TeacherTrainer:
    """
    Trainer for fine-tuning the teacher model using knowledge distillation from student models.
    Supports both standard fine-tuning and LoRA-based parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        config: Dict,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir: str = "./teacher_checkpoints"
    ):
        """
        Initialize the TeacherTrainer.
        
        Args:
            teacher_model: The teacher model to be fine-tuned
            config: Configuration dictionary containing training parameters
            device: Device to run training on (default: cuda if available, else cpu)
            output_dir: Directory for logs (checkpoints are not saved)
        """
        self.teacher_model = teacher_model
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LoRA if enabled
        if config.get("apply_lora_to_teacher", False):
            self._setup_lora()
        
        # Move model to device
        self.teacher_model = self.teacher_model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)
        
        # Loss functions
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.task_loss_fn = nn.MSELoss()  # Default for regression, can be overridden
        
        # Training state
        self.global_step = 0
        self.best_metric = -float('inf')
        self.latest_metrics = {}
        
        logger.info(f"Teacher trainer initialized on device: {self.device}")
        logger.info(f"Using LoRA: {hasattr(self, 'lora_config')}")
        logger.info("Checkpoint saving is disabled - model weights won't be saved to disk")
    
    def _setup_lora(self):
        """Initialize LoRA for the teacher model if enabled in config."""
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 16),
            lora_alpha=self.config.get("lora_alpha", 32.0),
            target_modules=self.config.get("lora_target_modules", ["query", "value"]),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias=self.config.get("lora_bias", "none"),
            task_type="FEATURE_EXTRACTION",
            modules_to_save=self.config.get("lora_modules_to_save", ["classifier"])
        )
        
        self.teacher_model = get_peft_model(self.teacher_model, lora_config)
        self.teacher_model.print_trainable_parameters()
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer for training."""
        # Get parameters that require gradients
        params = [p for p in self.teacher_model.parameters() if p.requires_grad]
        
        # Use separate learning rate for teacher if specified
        lr = self.config.get("teacher_learning_rate", self.config.get("learning_rate", 2e-5))
        
        return AdamW(
            params,
            lr=lr,
            weight_decay=self.config.get("weight_decay", 0.01),
            eps=self.config.get("adam_epsilon", 1e-8)
        )
    
    def _create_scheduler(self, optimizer: Optimizer):
        """Create learning rate scheduler."""
        num_warmup_steps = self.config.get("warmup_steps", 0)
        num_training_steps = self.config.get("max_steps", 1000)
        
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        student_model: Optional[nn.Module] = None,
        kd_alpha: float = 0.5,
        temperature: float = 2.0
    ) -> Dict[str, float]:
        """
        Train the teacher model for one epoch using knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            student_model: Student model for knowledge distillation (optional)
            kd_alpha: Weight for knowledge distillation loss
            temperature: Temperature for softening probability distributions
            
        Returns:
            Dictionary of training metrics
        """
        self.teacher_model.train()
        if student_model is not None:
            student_model.eval()
        
        total_loss = 0.0
        total_kd_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with torch.set_grad_enabled(True):
                teacher_outputs = self.teacher_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    output_hidden_states=True
                )
                
                # Calculate task loss
                task_loss = self.task_loss_fn(
                    teacher_outputs.logits.squeeze(),
                    batch['labels'].float()
                )
                
                # Knowledge distillation loss
                kd_loss = 0.0
                if student_model is not None and kd_alpha > 0:
                    with torch.no_grad():
                        student_outputs = student_model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            output_hidden_states=True
                        )
                    
                    # Softmax temperature scaling
                    teacher_logits = teacher_outputs.logits / temperature
                    student_logits = student_outputs.logits.detach() / temperature
                    
                    # KL divergence loss
                    kd_loss = self.kd_loss_fn(
                        F.log_softmax(teacher_logits, dim=-1),
                        F.softmax(student_logits, dim=-1)
                    ) * (temperature ** 2)  # Scale by temperature squared
                
                # Combined loss
                loss = (1 - kd_alpha) * task_loss + kd_alpha * kd_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get("max_grad_norm", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.teacher_model.parameters(),
                        self.config.get("max_grad_norm", 1.0)
                    )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                total_task_loss += task_loss.item()
                if kd_alpha > 0:
                    total_kd_loss += kd_loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'task_loss': f"{task_loss.item():.4f}",
                    'kd_loss': f"{kd_loss.item():.4f}" if kd_alpha > 0 else '0.0'
                })
        
        # Calculate epoch metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_task_loss': total_task_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if kd_alpha > 0:
            metrics['train_kd_loss'] = total_kd_loss / num_batches
        
        return metrics
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        metric_fn = None,
        metric_name: str = "eval_metric"
    ) -> Dict[str, float]:
        """
        Evaluate the teacher model on the evaluation dataset.
        
        Args:
            eval_loader: DataLoader for evaluation data
            metric_fn: Function to compute evaluation metric
            metric_name: Name of the evaluation metric
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.teacher_model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.teacher_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                loss = self.task_loss_fn(
                    outputs.logits.squeeze(),
                    batch['labels'].float()
                )
                
                # Update metrics
                total_loss += loss.item()
                all_predictions.append(outputs.logits.detach().cpu())
                all_labels.append(batch['labels'].detach().cpu())
        
        # Concatenate predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate metrics
        metrics = {
            'eval_loss': total_loss / len(eval_loader)
        }
        
        # Add custom metric if provided
        if metric_fn is not None:
            metrics[metric_name] = metric_fn(all_predictions, all_labels)
        
        return metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """
        Placeholder method - checkpoints are not saved to disk.
        This is kept for backward compatibility but performs no action.
        """
        logger.debug("Checkpoint saving is disabled in this configuration")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Placeholder method - no checkpoints are loaded from disk.
        This is kept for backward compatibility but performs no action.
        """
        logger.debug("Checkpoint loading is disabled in this configuration")
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int = 10,
        student_model: Optional[nn.Module] = None,
        kd_alpha: float = 0.5,
        temperature: float = 2.0,
        eval_metric_fn = None,
        metric_name: str = "eval_metric",
        early_stopping_patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Train the teacher model with knowledge distillation.
        
        Args:
            train_loader: DataLoader for training data
            eval_loader: DataLoader for evaluation data
            num_epochs: Number of training epochs
            student_model: Student model for knowledge distillation (optional)
            kd_alpha: Weight for knowledge distillation loss
            temperature: Temperature for softening probability distributions
            eval_metric_fn: Function to compute evaluation metric
            metric_name: Name of the evaluation metric
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary of training and evaluation metrics
        """
        # Initialize metrics tracking
        metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            metric_name: []
        }
        
        if kd_alpha > 0:
            metrics_history['train_kd_loss'] = []
            metrics_history['train_task_loss'] = []
        
        # Early stopping
        best_metric = -float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(
                train_loader,
                student_model=student_model,
                kd_alpha=kd_alpha,
                temperature=temperature
            )
            
            # Evaluate on validation set
            eval_metrics = self.evaluate(
                eval_loader,
                metric_fn=eval_metric_fn,
                metric_name=metric_name
            )
            
            # Update metrics history
            for key, value in {**train_metrics, **eval_metrics}.items():
                if key in metrics_history:
                    metrics_history[key].append(value)
            
            # Log metrics
            logger.info(f"Epoch {epoch} - " + 
                       f"Train Loss: {train_metrics['train_loss']:.4f}, " +
                       f"Eval Loss: {eval_metrics['eval_loss']:.4f}, " +
                       f"{metric_name}: {eval_metrics.get(metric_name, 'N/A'):.4f}")
            
            # Track best metric without saving checkpoints
            current_metric = eval_metrics.get(metric_name, -eval_metrics['eval_loss'])
            if current_metric > best_metric:
                best_metric = current_metric
                self.best_metric = best_metric
                patience_counter = 0
                logger.info(f"New best {metric_name}: {best_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement in {metric_name} for {patience_counter}/{early_stopping_patience} epochs")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Store the latest metrics for the server to access
        self.latest_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'eval_loss': eval_metrics['eval_loss'],
            metric_name: eval_metrics.get(metric_name, 0.0),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics_history
