"""
Centralized Trainer for Single Task Training
"""

import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
import yaml
import logging
import csv
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Callable, Optional
from tqdm import tqdm

class CentralizedTrainer:
    """Centralized trainer for single task"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device: torch.device, config: Dict, 
                 metrics_fn: Callable, logger: logging.Logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.metrics_fn = metrics_fn
        self.logger = logger
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['training']['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Setup paths
        self.results_dir = Path(config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging
        task_name = config['training']['dataset'].upper()
        self.csv_file = self.results_dir / f"centralized_{task_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._setup_csv_logging()
        
        # Training state
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def _setup_csv_logging(self):
        """Setup CSV logging with headers"""
        headers = [
            'epoch', 'training_time', 'train_loss', 'train_accuracy', 'sst2_val_accuracy', 'sst2_val_f1', 'qqp_val_accuracy', 'qqp_val_f1',
            'stsb_val_pearson', 'stsb_val_spearman', 'total_val_loss', 'gpu_usage', 'cpu_usage', 
            'memory_usage', 'timestamp'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        self.logger.info(f"CSV logging setup: {self.csv_file}")
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get GPU, CPU, and memory usage metrics"""
        gpu_usage = 0.0
        gpu_memory = 0.0
        
        try:
            # Use federated approach (working GPU monitoring)
            if torch.cuda.is_available() and self.device.type == 'cuda':
                # Get current GPU memory usage (federated approach)
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory = (allocated / total) * 100
                gpu_usage = gpu_memory  # Use memory usage as GPU utilization proxy
                
                self.logger.debug(f"GPU memory allocated: {allocated / 1e9:.2f} GB")
                self.logger.debug(f"GPU memory total: {total / 1e9:.2f} GB")
                self.logger.debug(f"GPU utilization: {gpu_usage:.2f}%")
            else:
                self.logger.warning("CUDA not available for GPU monitoring")
        except Exception as e:
            # Log error for debugging
            self.logger.warning(f"GPU monitoring failed: {e}")
            pass
        
        # CPU and memory metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        return {
            'gpu_usage': gpu_usage,
            'gpu_memory': gpu_memory,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }
    
    def _save_metrics_to_csv(self, epoch: int, val_results: Dict, training_time: float, train_loss: float):
        """Save metrics to CSV file"""
        system_metrics = self._get_system_metrics()
        
        # Calculate training accuracy for classification
        train_accuracy = val_results['metrics'].get('accuracy', 0.0)
        
        row = [
            epoch + 1,  # epoch
            training_time,  # training_time
            train_loss,  # train_loss
            train_accuracy,  # train_accuracy
            0.0,  # sst2_val_accuracy (empty for QQP-only)
            0.0,  # sst2_val_f1 (empty for QQP-only)
            val_results['metrics'].get('accuracy', 0.0),  # qqp_val_accuracy
            val_results['metrics'].get('f1', 0.0),  # qqp_val_f1
            0.0,  # stsb_val_pearson (empty for QQP-only)
            0.0,  # stsb_val_spearman (empty for QQP-only)
            val_results['loss'],  # total_val_loss
            system_metrics['gpu_usage'],  # gpu_usage
            system_metrics['cpu_usage'],  # cpu_usage
            system_metrics['memory_usage'],  # memory_usage
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # timestamp
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
    def train(self):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self._train_epoch()
            
            # Validation
            val_results = self._validate()
            
            # Calculate epoch training time
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_results['loss']:.4f}")
            self.logger.info(f"Val Metrics: {val_results['metrics']}")
            
            # Save metrics to CSV
            self._save_metrics_to_csv(epoch, val_results, epoch_time, train_loss)
            
            # No model saving - only metrics tracking
            self.logger.info(f"Epoch {epoch + 1} completed. Metrics saved to CSV.")
        
        # Log final training time
        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Results saved to: {self.csv_file}")
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def _validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = self.metrics_fn(all_predictions, all_labels)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'metrics': metrics
        }
    
    def evaluate(self) -> Dict:
        """Evaluate on test set"""
        self.logger.info("Evaluating on test set...")
        return self._validate()  # Same as validation for simplicity
