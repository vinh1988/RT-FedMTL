#!/usr/bin/env python3
"""
Federated Learning Client Implementation
Handles local training, LoRA updates, and synchronization
"""

import asyncio
import json
import logging
import time
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime

from federated_config import FederatedConfig
from src.lora.federated_lora import LoRAFederatedModel
from src.knowledge_distillation.federated_knowledge_distillation import LocalKDEngine
from src.communication.federated_websockets import WebSocketClient, MessageProtocol
from src.synchronization.federated_synchronization import ClientModelSynchronizer
from src.datasets.federated_datasets import DatasetFactory, DatasetConfig

logger = logging.getLogger(__name__)

class FederatedClient:
    """Federated Learning Client with LoRA, KD, and synchronization support"""

    def __init__(self, client_id: str, tasks: List[str], config: FederatedConfig):
        self.client_id = client_id
        self.tasks = tasks
        self.config = config
        self.device = self.get_device()

        # Initialize models
        self.student_model = LoRAFederatedModel(
            base_model_name=config.client_model,
            tasks=tasks,
            lora_rank=config.lora_rank
        )
        # Move model to device
        self.student_model = self.student_model.to(self.device)

        # Initialize optimizer for training
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1, 
            gamma=0.9
        )

        # Initialize KD engine
        self.kd_engine = LocalKDEngine(self.student_model, tasks, config)

        # Initialize synchronization
        self.websocket_client = WebSocketClient(
            f"ws://localhost:{config.port}",
            client_id
        )
        self.model_synchronizer = ClientModelSynchronizer(
            self.student_model, self.websocket_client
        )

        # Initialize datasets
        self.dataset_handlers = self.initialize_dataset_handlers()

        # Setup logging
        self.setup_logging()

        # Training state
        self.is_training = False
        self.current_round = 0

    def get_device(self):
        """Get available device (GPU/CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'federated_client_{self.client_id}.log'),
                logging.StreamHandler()
            ]
        )

    def initialize_dataset_handlers(self) -> Dict[str, Any]:
        """Initialize dataset handlers for available tasks"""
        handlers = {}

        for task in self.tasks:
            if task in self.config.task_configs:
                config = self.config.task_configs[task]
                dataset_config = DatasetConfig(
                    task_name=task,
                    train_samples=config.get('train_samples'),
                    val_samples=config.get('val_samples'),
                    random_seed=config.get('random_seed', 42)
                )
                handlers[task] = DatasetFactory.create_handler(task, dataset_config)

        return handlers

    async def connect_and_register(self):
        """Connect to server and register"""
        await self.websocket_client.connect()

        # Register with server
        registration_message = MessageProtocol.create_registration_message(
            self.client_id,
            self.tasks,
            {
                "model": self.config.client_model,
                "lora_rank": self.config.lora_rank,
                "supported_tasks": self.tasks
            }
        )

        await self.websocket_client.send(registration_message)
        logger.info(f"Client {self.client_id} registered with tasks: {self.tasks}")

    async def run_client(self):
        """Main client execution loop"""
        try:
            # Connect and register
            await self.connect_and_register()

            # Setup message handlers
            self.setup_message_handlers()

            # Keep connection alive
            while self.websocket_client.is_connected:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Client shutting down...")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            await self.websocket_client.disconnect()

    def setup_message_handlers(self):
        """Setup WebSocket message handlers"""
        self.websocket_client.register_message_handler(
            "global_model_sync", self.handle_global_model_sync
        )
        self.websocket_client.register_message_handler(
            "training_request", self.handle_training_request
        )
        self.websocket_client.register_message_handler(
            "registration_ack", self.handle_registration_ack
        )
        self.websocket_client.register_message_handler(
            "heartbeat", self.handle_heartbeat
        )

    async def handle_global_model_sync(self, data: Dict):
        """Handle incoming global model synchronization"""
        logger.info(f"Received global model synchronization")

        # Update local model with global knowledge
        sync_result = await self.model_synchronizer.synchronize_with_global_model(
            data["global_model_state"]
        )

        # Send acknowledgment
        await self.model_synchronizer.send_synchronization_acknowledgment(sync_result)

    async def handle_registration_ack(self, data: Dict):
        """Handle registration acknowledgment from server"""
        accepted = data.get("accepted", False)
        message = data.get("message", "")

        if accepted:
            logger.info(f"Registration acknowledged: {message}")
        else:
            logger.warning(f"Registration rejected: {message}")

    async def handle_heartbeat(self, data: Dict):
        """Handle heartbeat messages to keep connection alive"""
        # Send heartbeat response to keep connection alive
        heartbeat_response = MessageProtocol.create_heartbeat_message(self.client_id)
        await self.websocket_client.send(heartbeat_response)

    async def handle_training_request(self, data: Dict):
        """Handle training request from server"""
        round_num = data["round"]
        teacher_knowledge = data.get("teacher_knowledge", {})
        global_params = data.get("global_params", {})

        logger.info(f"Received training request for round {round_num}")
        logger.info(f"Starting training for client {self.client_id} with tasks: {self.tasks}")

        # Update teacher knowledge for KD
        if teacher_knowledge:
            # Convert lists back to tensors for teacher knowledge
            tensor_teacher_knowledge = {}
            for task, logits_list in teacher_knowledge.items():
                if isinstance(logits_list, list):
                    tensor_teacher_knowledge[task] = torch.tensor(logits_list, device=self.device)
                else:
                    tensor_teacher_knowledge[task] = logits_list
            self.kd_engine.update_teacher_knowledge(tensor_teacher_knowledge)

        # Perform local training
        self.is_training = True
        self.current_round = round_num

        try:
            local_metrics = await self.perform_local_training()

            # Extract LoRA updates
            lora_updates = self.student_model.get_all_lora_params()

            # Prepare student knowledge for reverse KD
            student_knowledge = self.kd_engine.prepare_student_knowledge_for_teacher(
                self.get_task_data_for_kd()
            )

            # Send update to server
            update_message = MessageProtocol.create_client_update_message(
                self.client_id,
                round_num,
                lora_updates,
                local_metrics,
                student_knowledge
            )

            # Use retry logic for sending updates
            logger.info(f"Attempting to send update to server for round {round_num}")
            success = await self.websocket_client.send(update_message, max_retries=5)
            if success:
                logger.info(f"Training completed and update sent for round {round_num}")
                logger.info(f"Client {self.client_id} metrics: {local_metrics}")
                logger.info(f"[SUCCESS] Training completed and update sent for round {round_num}")
                logger.info(f"[METRICS] Client {self.client_id} metrics: {local_metrics}")
            else:
                logger.error(f"Failed to send update for round {round_num} after retries")
                logger.error(f"WebSocket connection status: {self.websocket_client.is_connected}")
                logger.error(f"Update message size: {len(str(update_message))} characters")

        except Exception as e:
            logger.error(f"Error in local training for round {round_num}: {e}")
        finally:
            self.is_training = False

    async def perform_local_training(self) -> Dict[str, float]:
        """Perform local training with KD"""
        local_metrics = {}

        for task in self.tasks:
            if task in self.dataset_handlers:
                # Get data for this task
                dataset = self.dataset_handlers[task]
                task_data = dataset.prepare_data()

                # Train on this task with KD
                task_metrics = await self.train_task_with_kd(task, task_data)
                local_metrics[task] = task_metrics

        return local_metrics

    async def train_task_with_kd(self, task: str, task_data: Dict) -> Dict[str, float]:
        """Train on a specific task with KD"""
        
        # Split data into training and validation
        # Split data into training and validation
        # Dataset handler returns: texts/labels (train) and val_texts/val_labels (validation)
        train_data = {
            'texts': task_data.get('texts', []),
            'labels': task_data.get('labels', [])
        }
        val_data = {
            'texts': task_data.get('val_texts', []),
            'labels': task_data.get('val_labels', [])
        }
        
        # Get dataloaders
        train_dataloader = self.student_model.get_task_dataloader(
            task, self.config.batch_size, dataset_data=train_data
        )
        val_dataloader = None
        if val_data['texts'] and val_data['labels']:
            val_dataloader = self.student_model.get_task_dataloader(
                task, self.config.batch_size, dataset_data=val_data
            )
            logger.info(f"Validation dataloader created for task {task} with batch_size {self.config.batch_size}")
            logger.info(f"[VALIDATION] Validation dataloader created for task {task} with batch_size {self.config.batch_size}")

        logger.info(f"Training dataloader created for task {task} with batch_size {self.config.batch_size}")
        logger.info(f"[TRAINING] Training dataloader created for task {task} with batch_size {self.config.batch_size}")

        # Set model to training mode
        self.student_model.train()
        
        logger.info(f"Starting training for task {task} with {len(train_dataloader)} batches")
        logger.info(f"[TRAINING] Starting training for task {task} with {len(train_dataloader)} batches")
        if val_dataloader:
            logger.info(f"Starting validation for task {task} with {len(val_dataloader)} batches")
            logger.info(f"[VALIDATION] Starting validation for task {task} with {len(val_dataloader)} batches")

        # Training loop with proper metrics calculation
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0

        for batch in train_dataloader:
            # Unpack batch tuple (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = batch

            # Move tensors to the correct device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Add check for empty or scalar batches
            if len(input_ids) == 0 or input_ids.dim() == 0:
                logger.warning(f"Skipping empty or scalar batch for task {task}")
                continue

            # Ensure labels are not scalars
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.student_model(
                input_ids,
                attention_mask,
                task
            )

            # Calculate KD loss (IMPROVED: pass current_round for progressive KD)
            kd_loss = self.kd_engine.calculate_kd_loss(
                logits, task, labels, current_round=self.current_round
            )

            # Backward pass
            kd_loss.backward()
            
            # PHASE 2: Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
                max_norm=1.0
            )
            
            # Update parameters
            self.optimizer.step()

            total_loss += kd_loss.item()
            num_batches += 1

            # Log progress every few batches
            if num_batches % 5 == 0:
                logger.info(f"Task {task} - Batch {num_batches}, Loss: {kd_loss.item():.4f}")
                logger.info(f"[STATS] Task {task} - Batch {num_batches}, Loss: {kd_loss.item():.4f}")

            # Calculate predictions and accuracy
            with torch.no_grad():
                if task == 'stsb':  # Regression task
                    predictions = logits.squeeze()
                    # For regression, use a tolerance-based accuracy
                    # Consider predictions "correct" if they're within 0.1 of the true label
                    if labels.dim() == 0:
                        labels_reshaped = labels.unsqueeze(0)
                    else:
                        labels_reshaped = labels
                    
                    # Ensure predictions are not scalars
                    if predictions.dim() == 0:
                        predictions = predictions.unsqueeze(0)
                    
                    # Calculate tolerance-based accuracy for regression
                    tolerance = 0.05  # Within 0.05 of true value (5% tolerance)
                    correct_predictions += (torch.abs(predictions - labels_reshaped) <= tolerance).sum().item()
                else:  # Classification tasks
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                
                total_samples += labels.size(0)
                
                # Only extend lists if predictions and labels are arrays, not scalars
                pred_cpu = predictions.cpu()
                if pred_cpu.numel() > 1:  # More than one element
                    all_predictions.extend(pred_cpu.numpy().flatten())
                else:
                    all_predictions.append(pred_cpu.item())
                
                label_cpu = labels.cpu()
                if label_cpu.numel() > 1:
                    all_labels.extend(label_cpu.numpy().flatten())
                else:
                    all_labels.append(label_cpu.item())

        # Update learning rate scheduler
        self.scheduler.step()

        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Task {task} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"[SUCCESS] Task {task} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Add regression-specific metrics for STSB
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }
        
        if task == 'stsb' and len(all_predictions) > 0 and len(all_labels) > 0:
            # Calculate additional regression metrics
            import numpy as np
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Ensure arrays are not empty and have proper shape
            if pred_array.size == 0 or label_array.size == 0:
                logger.warning(f"Empty prediction or label arrays for task {task}")
                pred_array = np.array([0.0])  # Fallback
                label_array = np.array([0.0])
            
            # Handle scalar arrays
            if pred_array.ndim == 0:
                pred_array = np.array([pred_array])
            if label_array.ndim == 0:
                label_array = np.array([label_array])
            
            # Mean Absolute Error
            mae = np.mean(np.abs(pred_array - label_array))
            # Mean Squared Error
            mse = np.mean((pred_array - label_array) ** 2)
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            
            # Correlation coefficient (handle edge cases)
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    correlation = np.corrcoef(pred_array, label_array)[0, 1]
                    if np.isnan(correlation) or np.isinf(correlation):
                        correlation = 0.0
                except (ValueError, IndexError):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # For regression, use correlation as the primary accuracy metric
            regression_accuracy = max(0, correlation)  # Clamp negative correlations to 0
            
            # Tolerance-based correct count
            tolerance = 0.1  # 10% tolerance
            tolerance_correct = np.sum(np.abs(pred_array - label_array) <= tolerance)
            
            logger.info(f"STSB Regression Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, Correlation: {correlation:.4f}")
            logger.info(f"[REGRESSION] STSB Regression Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, Correlation: {correlation:.4f}")
            
            metrics.update({
                'accuracy': float(regression_accuracy),
                'correct_predictions': int(tolerance_correct),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'correlation': float(correlation)
            })

        # Add validation metrics if validation data is available
        if val_dataloader is not None:
            val_metrics = self.evaluate_on_validation(task, val_dataloader)
            metrics['val_accuracy'] = val_metrics['accuracy']
            metrics['val_loss'] = val_metrics['loss']
            metrics['val_samples'] = val_metrics['samples_processed']
            metrics['val_correct_predictions'] = val_metrics['correct_predictions']
            if 'correlation' in val_metrics:
                metrics['val_correlation'] = val_metrics['correlation']
                metrics['val_mae'] = val_metrics['mae']
            logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"[VALIDATION] Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        return metrics

    def evaluate_on_validation(self, task: str, val_dataloader) -> Dict[str, float]:
        """Evaluate model on validation data"""
        import numpy as np
        
        # Set model to evaluation mode
        self.student_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch
                
                # Move tensors to the correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Add check for empty or scalar batches
                if len(input_ids) == 0 or input_ids.dim() == 0:
                    logger.warning(f"Skipping empty or scalar validation batch for task {task}")
                    continue

                # Ensure labels are not scalars
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Forward pass
                logits = self.student_model(input_ids, attention_mask, task)
                
                # Calculate loss (IMPROVED: pass current_round)
                loss = self.kd_engine.calculate_kd_loss(logits, task, labels, current_round=self.current_round)
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate predictions
                if task == 'stsb':  # Regression task
                    predictions = logits.squeeze()
                    tolerance = 0.1
                    if labels.dim() == 0:
                        labels_reshaped = labels.unsqueeze(0)
                    else:
                        labels_reshaped = labels
                    pred_reshaped = predictions.unsqueeze(0) if predictions.dim() == 0 else predictions
                    correct_predictions += (torch.abs(pred_reshaped - labels_reshaped) <= tolerance).sum().item()
                else:  # Classification tasks
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                
                total_samples += labels.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples_processed': total_samples,
            'correct_predictions': correct_predictions
        }
        
        # Add regression-specific metrics for STSB
        if task == 'stsb' and len(all_predictions) > 0 and len(all_labels) > 0:
            pred_array = np.array(all_predictions)
            label_array = np.array(all_labels)
            
            # Ensure arrays are not empty and have proper shape
            if pred_array.size == 0 or label_array.size == 0:
                logger.warning(f"Empty prediction or label arrays for validation task {task}")
                pred_array = np.array([0.0])  # Fallback
                label_array = np.array([0.0])
            
            # Handle scalar arrays
            if pred_array.ndim == 0:
                pred_array = np.array([pred_array])
            if label_array.ndim == 0:
                label_array = np.array([label_array])
            
            mae = np.mean(np.abs(pred_array - label_array))
            mse = np.mean((pred_array - label_array) ** 2)
            
            # Correlation coefficient
            if len(pred_array) > 1 and np.std(pred_array) > 0 and np.std(label_array) > 0:
                try:
                    correlation = np.corrcoef(pred_array, label_array)[0, 1]
                    if np.isnan(correlation) or np.isinf(correlation):
                        correlation = 0.0
                except (ValueError, IndexError):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            regression_accuracy = max(0, correlation)
            tolerance_correct = np.sum(np.abs(pred_array - label_array) <= 0.1)
            
            metrics.update({
                'accuracy': float(regression_accuracy),
                'correct_predictions': int(tolerance_correct),
                'mae': float(mae),
                'mse': float(mse),
                'correlation': float(correlation)
            })
        
        # Set model back to training mode
        self.student_model.train()
        
        return metrics

    def get_task_data_for_kd(self) -> Dict[str, Dict]:
        """Get task data for student knowledge preparation"""
        from transformers import AutoTokenizer
        
        task_data = {}
        tokenizer = AutoTokenizer.from_pretrained(self.config.client_model)

        for task in self.tasks:
            if task in self.dataset_handlers:
                dataset = self.dataset_handlers[task]
                data = dataset.prepare_data()
                
                # Tokenize the texts for KD
                texts = data.get('texts', [])
                if texts:
                    tokenized = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )
                    
                    # Add tokenized data to the task data
                    data['input_ids'] = tokenized['input_ids']
                    data['attention_mask'] = tokenized['attention_mask']
                
                task_data[task] = data

        return task_data

    def extract_lora_updates(self) -> Dict[str, Dict]:
        """Extract LoRA parameters for federated aggregation"""
        return self.student_model.get_all_lora_params()

    def get_client_status(self) -> Dict:
        """Get current client status"""
        return {
            "client_id": self.client_id,
            "tasks": self.tasks,
            "is_connected": self.websocket_client.is_connected,
            "is_training": self.is_training,
            "current_round": self.current_round,
            "model_synchronized": self.model_synchronizer.is_synchronized,
            "available_tasks": list(self.dataset_handlers.keys())
        }

def run_client(client_id: str, tasks: List[str], config: FederatedConfig):
    """Run a federated learning client"""
    client = FederatedClient(client_id, tasks, config)
    asyncio.run(client.run_client())

if __name__ == "__main__":
    import argparse
    from federated_config import create_argument_parser, load_config

    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments for client mode
    if args.mode != "client":
        parser.error("This script is for client mode only.")

    if not args.client_id:
        parser.error("Client ID is required for client mode.")

    if not args.tasks:
        parser.error("Tasks are required for client mode.")

    config = load_config(args)
    config.print_summary()

    run_client(args.client_id, args.tasks, config)
