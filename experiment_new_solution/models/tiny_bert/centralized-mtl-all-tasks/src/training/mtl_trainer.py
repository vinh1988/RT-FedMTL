"""
Centralized Multi-Task Learning Trainer
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
from typing import Dict, Callable
from tqdm import tqdm

class CentralizedMTLTrainer:
    """Centralized trainer for multi-task learning"""
    
    def __init__(self, model, train_loaders: Dict, val_loaders: Dict, test_loaders: Dict,
                 device: torch.device, config: Dict, metrics_fns: Dict, logger: logging.Logger):
        self.model = model
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.device = device
        self.config = config
        self.metrics_fns = metrics_fns
        self.logger = logger
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Calculate total steps for scheduler
        total_steps = 0
        for task_name, loader in train_loaders.items():
            total_steps += len(loader)
        total_steps *= config['training']['epochs']
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Setup paths
        self.results_dir = Path(config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging
        self.csv_file = self.results_dir / f"centralized_mtl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._setup_csv_logging()
        
        # Setup training CSV files for each task
        self.training_csv_files = {}
        for task_name in train_loaders.keys():
            task_csv_file = self.results_dir / f"training_{task_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self._setup_training_csv(task_name, task_csv_file)
            self.training_csv_files[task_name] = task_csv_file
        
        # Training state
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def _setup_csv_logging(self):
        """Setup CSV logging with headers matching federated format"""
        headers = [
            'epoch', 'training_time', 'sst2_val_accuracy', 'sst2_val_f1', 
            'qqp_val_accuracy', 'qqp_val_f1', 'stsb_val_pearson', 'stsb_val_spearman',
            'total_val_loss', 'gpu_usage', 'cpu_usage', 'memory_usage', 'timestamp'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        self.logger.info(f"CSV logging setup: {self.csv_file}")
    
    def _setup_training_csv(self, task_name: str, task_csv_file: Path):
        """Setup training CSV for specific task"""
        headers = ['epoch', 'train_loss', 'train_accuracy', 'timestamp']
        
        with open(task_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        self.logger.info(f"Training CSV setup for {task_name}: {task_csv_file}")
    
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
    
    def _save_metrics_to_csv(self, epoch: int, val_results: Dict, training_time: float):
        """Save metrics to CSV file"""
        system_metrics = self._get_system_metrics()
        
        # Extract metrics from validation results
        row = [
            epoch + 1,  # epoch
            training_time,  # training_time
            val_results.get('sst2', {}).get('metrics', {}).get('accuracy', 0.0),  # sst2_val_accuracy
            val_results.get('sst2', {}).get('metrics', {}).get('f1', 0.0),  # sst2_val_f1
            val_results.get('qqp', {}).get('metrics', {}).get('accuracy', 0.0),  # qqp_val_accuracy
            val_results.get('qqp', {}).get('metrics', {}).get('f1', 0.0),  # qqp_val_f1
            val_results.get('stsb', {}).get('metrics', {}).get('pearson', 0.0),  # stsb_val_pearson
            val_results.get('stsb', {}).get('metrics', {}).get('spearman', 0.0),  # stsb_val_spearman
            val_results.get('total_loss', 0.0),  # total_val_loss
            system_metrics['gpu_usage'],  # gpu_usage
            system_metrics['cpu_usage'],  # cpu_usage
            system_metrics['memory_usage'],  # memory_usage
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # timestamp
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _save_training_metrics(self, task_name: str, epoch: int, train_loss: float, train_accuracy: float):
        """Save training metrics to task-specific CSV"""
        row = [
            epoch + 1,  # epoch
            train_loss,  # train_loss
            train_accuracy,  # train_accuracy
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # timestamp
        ]
        
        with open(self.training_csv_files[task_name], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
    def train(self):
        """Main training loop"""
        epochs = self.config['training']['epochs']
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            self.current_epoch = epoch  # Track current epoch for training metrics
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_results = self._train_epoch()
            
            # Validation
            val_results = self._validate()
            
            # Calculate epoch training time
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # Log results
            self.logger.info(f"Train Results: {train_results}")
            self.logger.info(f"Val Results: {val_results}")
            
            # Save metrics to CSV
            self._save_metrics_to_csv(epoch, val_results, epoch_time)
            
            # No model saving - only metrics tracking
            self.logger.info(f"Epoch {epoch + 1} completed. Metrics saved to CSV.")
        
        # Log final training time
        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Results saved to: {self.csv_file}")
    
    def _train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_results = {}
        total_loss = 0
        
        for task_name, loader in self.train_loaders.items():
            task_loss = 0
            task_accuracy = 0
            batch_count = 0
            progress_bar = tqdm(loader, desc=f"Training {task_name.upper()}")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    task_name=task_name, 
                    labels=labels
                )
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                
                task_loss += loss.item()
                self.global_step += 1
                batch_count += 1
                
                # Calculate batch accuracy
                with torch.no_grad():
                    logits = outputs['logits']
                    if self.model.task_configs[task_name]['task_type'] == 'classification':
                        predictions = torch.argmax(logits, dim=-1)
                        correct = (predictions == labels).sum().item()
                        total = labels.size(0)
                        batch_accuracy = correct / total
                    else:  # regression
                        predictions = logits.squeeze()
                        # For regression, use R² as accuracy metric
                        ss_res = ((labels - predictions) ** 2).sum()
                        ss_tot = ((labels - labels.mean()) ** 2).sum()
                        batch_accuracy = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    task_accuracy += batch_accuracy
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item(), 'acc': f'{batch_accuracy:.4f}'})
            
            # Save epoch-level training metrics
            avg_train_loss = task_loss / batch_count
            avg_train_accuracy = task_accuracy / batch_count
            self._save_training_metrics(
                task_name=task_name,
                epoch=self.current_epoch if hasattr(self, 'current_epoch') else 0,
                train_loss=avg_train_loss,
                train_accuracy=avg_train_accuracy
            )
            
            epoch_results[task_name] = avg_train_loss
            total_loss += task_loss
        
        epoch_results['total_loss'] = total_loss / len(self.train_loaders)
        return epoch_results
    
    def _validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        val_results = {}
        total_loss = 0
        
        with torch.no_grad():
            for task_name, loader in self.val_loaders.items():
                task_loss = 0
                all_predictions = []
                all_labels = []
                
                for batch in tqdm(loader, desc=f"Validating {task_name.upper()}"):
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        task_name=task_name, 
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    task_loss += loss.item()
                    
                    # Collect predictions
                    logits = outputs['logits']
                    if self.model.task_configs[task_name]['task_type'] == 'classification':
                        predictions = torch.argmax(logits, dim=-1)
                    else:  # regression
                        predictions = logits.squeeze()
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                # Compute metrics
                metrics = self.metrics_fns[task_name](all_predictions, all_labels)
                
                val_results[task_name] = {
                    'loss': task_loss / len(loader),
                    'metrics': metrics
                }
                total_loss += task_loss
        
        val_results['total_loss'] = total_loss / len(self.val_loaders)
        return val_results
    
    def evaluate(self) -> Dict:
        """Evaluate on test sets"""
        self.logger.info("Evaluating on test sets...")
        return self._validate()  # Same as validation for simplicity
